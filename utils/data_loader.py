# -*- coding: utf-8 -*-
import json
import random
from collections import defaultdict
import logging
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from config import DATA_CONFIG, SL_CONFIG

logger = logging.getLogger(__name__)

class RecDataset(Dataset):
    """
    A PyTorch Dataset class for preparing data for the DSR-Rec model.
    
    Functionality:
    1. Reads and parses the raw JSON data.
    2. Maps `user_id` and `item_id` to contiguous integers starting from 0.
    3. Constructs interaction sequences for each user based on timestamps.
    4. Implements the Stochastic Length (SL) historical sequence preprocessing algorithm proposed in the paper.
    5. Provides a `__getitem__` method to be called by the DataLoader.
    """
    def __init__(self, file_path, min_seq_len):
        super().__init__()
        self.file_path = file_path
        self.min_seq_len = min_seq_len
        
        # Hyperparameters for the SL algorithm
        self.sl_nc = SL_CONFIG['max_len_nc']
        self.sl_alpha = SL_CONFIG['alpha']
        self.sl_l_sub = round(self.sl_nc ** (self.sl_alpha / 2))
        
        # Initialize data containers
        self.user_sequences = []
        self.user2id = {}
        self.item2id = {}
        self.id2user = {}
        self.id2item = {}
        self.item_meta_data = {} # Store item text metadata

        self._load_and_process_data()

    def _get_or_add_id(self, mapping, reverse_mapping, key):
        if key not in mapping:
            new_id = len(mapping)
            mapping[key] = new_id
            reverse_mapping[new_id] = key
        return mapping[key]

    def _load_and_process_data(self):
        df = pd.read_json(self.file_path, lines=True)
        # Sort by user ID and timestamp to ensure correct interaction sequence order
        df = df.sort_values(by=['reviewerID', 'unixReviewTime'], ascending=True)

        logger.info("Data loaded, starting to build user sequences...")

        # Group by user to build sequences
        grouped = df.groupby('reviewerID')
        
        temp_sequences = []
        for reviewerID, group in tqdm(grouped, desc="Processing users"):
            # Filter out users with too few interactions
            if len(group) < self.min_seq_len:
                continue

            user_id = self._get_or_add_id(self.user2id, self.id2user, reviewerID)
            
            seq = []
            for _, row in group.iterrows():
                asin = row['asin']
                item_id = self._get_or_add_id(self.item2id, self.id2item, asin)
                seq.append(item_id)
                
                if item_id not in self.item_meta_data:
                    title = str(row.get('title', ''))
                    category = str(row.get('category', ''))
                    description = str(row.get('description', ''))
                    # Concatenate fields to create the item's text information.
                    self.item_meta_data[item_id] = f"{title},{category},{description}"

            temp_sequences.append((user_id, seq))
        
        self.user_sequences = temp_sequences


    @property
    def user_count(self):
        return len(self.user2id)

    @property
    def item_count(self):
        return len(self.item2id)

    def __len__(self):
        return len(self.user_sequences)

    def __getitem__(self, index):
        """
        Gets a single sample, containing the processed history sequence and the target item.
        The SL algorithm is implemented here.
        """
        user_id, item_sequence = self.user_sequences[index]
        
        # Sequence length must be at least 2 to be split into history and target
        if len(item_sequence) < 2:
            return None 

        history = item_sequence[:-1]
        target = item_sequence[-1]
        
        # --- Stochastic Length (SL) algorithm implementation ---
        processed_history = self._apply_sl(history)

        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'history': torch.tensor(processed_history, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }

    def _apply_sl(self, sequence):
        """
        Applies the Stochastic Length algorithm to an input sequence.

        Args:
            sequence (list): The user's historical interaction sequence.

        Returns:
            list: The processed historical interaction sequence.
        """
        t_u = len(sequence)
        
        # If the original sequence length is less than or equal to the subsampling length threshold,
        # use the original sequence directly.
        if t_u <= self.sl_l_sub:
            return sequence
            
        # Calculate the probability of using the full sequence
        p_use_full = min(1.0, (self.sl_nc ** self.sl_alpha) / (t_u ** 2))
        
        if random.random() < p_use_full:
            # With probability p_use_full, use the full sequence
            return sequence
        else:
            # Otherwise, randomly sample a subsequence of length L_sub
            # To preserve temporality, we randomly choose from all possible starting points
            start_index = random.randint(0, t_u - self.sl_l_sub)
            return sequence[start_index : start_index + self.sl_l_sub]

def collate_fn(batch):
    """
    To handle variable length sequences returned by `__getitem__`.
    Pads all history sequences in a batch to the length of the longest sequence in that batch.
    """
    # Filter out any potential None items
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    batch.sort(key=lambda x: len(x['history']), reverse=True)

    user_ids = torch.stack([item['user_id'] for item in batch])
    histories = [item['history'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    
    # Pad the history sequences
    # Use 0 as the padding_value
    padded_histories = torch.nn.utils.rnn.pad_sequence(histories, batch_first=True, padding_value=0) 

    return {
        'user_id': user_ids,
        'history': padded_histories,
        'target': targets
    }