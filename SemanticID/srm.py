# coding:utf-8
import torch
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import numpy as np
import logging
from tqdm import tqdm
from torch.nn import functional as F
logger = logging.getLogger(__name__)

def semantic_redirection_matching(
    conflict_groups: dict, 
    codebook_l: torch.Tensor, 
    base_id_map: dict
):
    """
    Implements the Semantic Redirection Matching (SRM) algorithm to resolve semantic ID conflicts.
    
    Args:
        conflict_groups (dict): The conflict groups.
        Format: {base_id_tuple: [{'item_id': int, 'residual': torch.Tensor}, ...]}

        codebook_l (torch.Tensor): The codebook of the last layer (L-th layer) of the RQ-VAE.
        shape: (codebook_size, embedding_dim)

        base_id_map (dict): The set of already assigned base semantic IDs, used for finding available candidate IDs.
        Format: {base_id_tuple: item_id}
            
    Returns:
        dict: The updated mapping from item IDs to unique semantic IDs.
        Format: {item_id: unique_id_tuple}
    """
    
    final_item_to_semantic_id = {}
    
    # 1. puts all items without conflicts into the final mapping
    non_conflict_items = {item_id: base_id for base_id, item_id in base_id_map.items() if base_id not in conflict_groups}
    final_item_to_semantic_id.update(non_conflict_items)
    
    codebook_size, embedding_dim = codebook_l.shape
    device = codebook_l.device
    
    # 2. Identify all ID indices that are already occupied in the final layer.
    # All the ids used in the last layer
    used_l_layer_indices = {base_id[-1] for base_id in base_id_map.keys()}
    
    # 3. Iterate through each conflict group to resolve its conflicts.
    for base_id, items in tqdm(conflict_groups.items(), desc="SRM"):
        conflict_source_index = base_id[-1]
        v_source = codebook_l[conflict_source_index] # conflict source vector
        
        # The pool of available candidate IDs.
        # The candidate pool must not contain any indices already used by other base_ids in the final layer.
        available_candidate_indices = [
            j for j in range(codebook_size) if j not in used_l_layer_indices
        ]
        if len(available_candidate_indices) < len(items):
            items_to_resolve = items[:len(available_candidate_indices)]
            items_unresolved = items[len(available_candidate_indices):]
        else:
            items_to_resolve = items
            items_unresolved = []

        if not items_to_resolve:
            for item in items_unresolved:
                 final_item_to_semantic_id[item['item_id']] = base_id 
            continue
            
        num_items_to_resolve = len(items_to_resolve)
        num_candidates = len(available_candidate_indices)
            
        # calculate the Fit Score matrix (Equation 2)
        # fit_matrix[i, j]  represents the fitness score between the i-th conflicting item and the j-th candidate ID.
        fit_matrix = torch.zeros((num_items_to_resolve, num_candidates), device=device)
        
        residuals = torch.stack([item['residual'] for item in items_to_resolve]) # (num_items, D)
        candidate_vectors = codebook_l[available_candidate_indices] # (num_candidates, D)
        
        # This is based on the deviation of the candidate vector from the conflict source's vector. 
        deviation_vectors = candidate_vectors - v_source # (num_candidates, D)
        
        # calculate cosine similarity
        cos_sim = F.cosine_similarity(residuals.unsqueeze(1), deviation_vectors.unsqueeze(0), dim=2)
        fit_matrix = cos_sim # shape: (num_items, num_candidates)
        
        # using Hungarian algorithm
 
        cost_matrix = -fit_matrix.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # update semantic ID
        prefix_id = base_id[:-1]
        for i, j in zip(row_ind, col_ind):
            item_id = items_to_resolve[i]['item_id']
            new_l_layer_index = available_candidate_indices[j]
            new_semantic_id = prefix_id + (new_l_layer_index,)
            
            final_item_to_semantic_id[item_id] = new_semantic_id
            
            # Remove the newly assigned IDs from the available candidate pool to prevent reuse within the same loop.
            used_l_layer_indices.add(new_l_layer_index)

        # Handle items that remain unresolved due to an insufficient candidate pool.
        for item in items_unresolved:
            final_item_to_semantic_id[item['item_id']] = base_id # maintain the original conflict ID.

    logger.info(f"SRM completed.processed {len(conflict_groups)} conflict groups.")
    return final_item_to_semantic_id


