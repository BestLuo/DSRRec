# coding:utf-8
import torch
import os
import pickle
import json
from collections import defaultdict
from tqdm import tqdm
import logging

from config import PROJECT_ROOT, QWEN3_EMBEDDING_CONFIG, RQ_VAE_CONFIG
from qwen3_encoder import Qwen3Embedding
from rq_vae import RQVAE, train_rqvae
from data_loader import RecDataset
from srm import semantic_redirection_matching

logger = logging.getLogger(__name__)

class SemanticIDGenerator:
    """
    Generate a Unique Semantic ID for All Items.
    This is an offline processing workflow, divided into three main steps:
    1. Utilize the Qwen3-Embedding model to generate text vectors for the metadata of all items.
    2. Train an RQ-VAE model to quantize the text vectors into base semantic IDs and residuals.
    3. Apply the SRM algorithm to resolve conflicts among the base semantic IDs, thereby generating the final unique ID mapping.
    """
    def __init__(self, dataset, qwen3_encoder, rq_vae_model, device):
        self.dataset = dataset
        self.qwen3_encoder = qwen3_encoder
        self.rq_vae_model = rq_vae_model
        self.device = device

    def generate_item_embeddings(self, batch_size=8):
        """
        Step 1: Generate text embeddings for all items in the dataset.
        """
        # 1. Retrieve and sort all raw string IDs to ensure a consistent processing order.
        sorted_string_item_ids = sorted(self.dataset.item2id.keys())
        
        # 2. Prepare the list of item text descriptions in the corresponding order.
        item_texts = []
        for string_id in sorted_string_item_ids:
            internal_int_id = self.dataset.item2id[string_id]
            item_texts.append(self.dataset.item_meta_data[internal_int_id])

        all_embeddings = self.qwen3_encoder.encode(
            item_texts, 
            dim=QWEN3_EMBEDDING_CONFIG['output_dim'],
            batch_size=batch_size,
            show_progress=True 
        )
        
        self.item_embeddings = torch.from_numpy(all_embeddings).to(self.device, dtype=torch.float32)
        
        self.item_id_map = {string_id: i for i, string_id in enumerate(sorted_string_item_ids)}

    def train_and_apply_rq_vae(self):
        """
        Step 2: Train an RQ-VAE and use it to quantize the item embeddings.
        """
        embedding_dataset = torch.utils.data.TensorDataset(self.item_embeddings)
        data_loader = torch.utils.data.DataLoader(
            embedding_dataset, 
            batch_size=RQ_VAE_CONFIG['batch_size'], 
            shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.rq_vae_model.parameters(), 
            lr=1e-3 
        )
        
        effective_epochs = 30 

        train_rqvae(self.rq_vae_model, data_loader, optimizer, effective_epochs, self.device)

        # Utilize the trained RQ-VAE for quantization.
        self.rq_vae_model.eval()
        with torch.no_grad():
            encoded_embeddings = self.rq_vae_model.encoder(self.item_embeddings)
            quantized_indices, final_residuals = self.rq_vae_model.quantize(encoded_embeddings)
        
        self.quantized_indices = quantized_indices.cpu().numpy()
        self.final_residuals = final_residuals.cpu()

    def resolve_conflicts_with_srm(self):
        """
        Step 3: Identify and resolve conflicts using SRM.
        """
        
        base_id_to_items = defaultdict(list)
        reverse_item_id_map = {v: k for k, v in self.item_id_map.items()}

        for i in range(len(self.quantized_indices)):
            item_id = reverse_item_id_map[i]
            base_id_tuple = tuple(self.quantized_indices[i])
            item_info = {'item_id': item_id, 'residual': self.final_residuals[i]}
            base_id_to_items[base_id_tuple].append(item_info)

        # 1. Initialize the final ID mapping table and populate it first with all non-conflicting items.
        self.final_item_to_semantic_id = {}
        for base_id_tuple, items_list in base_id_to_items.items():
            if len(items_list) == 1:
                item_info = items_list[0]
                self.final_item_to_semantic_id[item_info['item_id']] = list(base_id_tuple)
        # 2. Filter out the conflicting groups that require resolution.
        conflict_groups = {k: v for k, v in base_id_to_items.items() if len(v) > 1}


        # 3. If a conflict exists, execute the SRM algorithm.
        if conflict_groups:
            # Construct a map containing all currently known IDs for the SRM to check if an ID is already occupied.
            # The key is the tuple ID, and the value is the item's string ID.
            base_id_map_for_srm = {tuple(v): k for k, v in self.final_item_to_semantic_id.items()}
            # Add a placeholder for the conflicting ID.
            for base_id_tuple in conflict_groups:
                 base_id_map_for_srm[base_id_tuple] = -1 # Flag the item as the conflict source.

            codebook_l = self.rq_vae_model.codebooks[-1].weight.data.cpu()
            
            # Use the SRM algorithm to process only the conflicting items.
            resolved_conflicts = semantic_redirection_matching(
                conflict_groups=conflict_groups,
                codebook_l=codebook_l,
                base_id_map=base_id_map_for_srm
            )
            
            # 4. Update the final mapping table with the processed IDs of the conflicting items.
            self.final_item_to_semantic_id.update(resolved_conflicts)
            logger.info(f"SRM conflict resolution complete.")
        else:
            logger.info("No conflicts detected; SRM execution is not required.")
    

    def generate(self):
        """
        Executing the full semantic ID generation pipeline.
        """
        self.generate_item_embeddings()
        self.train_and_apply_rq_vae()
        self.resolve_conflicts_with_srm()
        return self.final_item_to_semantic_id

def get_semantic_ids(dataset, device):
    """
    Main function to retrieve or generate semantic IDs.
    """
    save_dir = os.path.join(PROJECT_ROOT, "generated_data")
    os.makedirs(save_dir, exist_ok=True)
    id_map_path_pkl = os.path.join(save_dir, "item_to_semantic_id.pkl")
    id_map_path_json = os.path.join(save_dir, "item_to_semantic_id.json")
    rq_vae_path = os.path.join(save_dir, "rq_vae_model.pth")

    if os.path.exists(id_map_path_pkl) and os.path.exists(rq_vae_path):
        with open(id_map_path_pkl, 'rb') as f:
            item_to_semantic_id = pickle.load(f)
        rq_vae_model = RQVAE(
            input_dim=RQ_VAE_CONFIG['input_dim'],
            output_dim=RQ_VAE_CONFIG['output_dim'],
            num_levels=RQ_VAE_CONFIG['num_levels'],
            codebook_size=RQ_VAE_CONFIG['codebook_size'],
            embedding_dim=RQ_VAE_CONFIG['embedding_dim']
        )
        rq_vae_model.load_state_dict(torch.load(rq_vae_path))
        return item_to_semantic_id, rq_vae_model
    
    # 1. Initialize the RQ-VAE model.
    qwen3_encoder = Qwen3Embedding(QWEN3_EMBEDDING_CONFIG['model_path'], use_cuda=device.startswith('cuda'))
    rq_vae_model = RQVAE(
        input_dim=RQ_VAE_CONFIG['input_dim'],
        output_dim=RQ_VAE_CONFIG['output_dim'],
        num_levels=RQ_VAE_CONFIG['num_levels'],
        codebook_size=RQ_VAE_CONFIG['codebook_size'],
        embedding_dim=RQ_VAE_CONFIG['embedding_dim']
    ).to(device)

    # 2. Create the semantic ID generator and execute semantic ID generation.
    generator = SemanticIDGenerator(dataset, qwen3_encoder, rq_vae_model, device)
    item_to_semantic_id = generator.generate()

    # 3. Save the generated semantic IDs.
    with open(id_map_path_pkl, 'wb') as f:
        pickle.dump(item_to_semantic_id, f)
    
    try:
        item_to_semantic_id_json = {str(k): v for k, v in item_to_semantic_id.items()}
        with open(id_map_path_json, 'w', encoding='utf-8') as f:
            json.dump(item_to_semantic_id_json, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"error: {e}")
    torch.save(rq_vae_model.state_dict(), rq_vae_path)

    return item_to_semantic_id, rq_vae_model






