import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import logging

from config import *
from utils import set_seed, setup_logger
from data_loader import RecDataset, collate_fn
from id_generator import get_semantic_ids
from trie import build_trie_from_id_map
from dsr_rec_model import DSRRec
from trainer import Trainer
import os
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

def main():
    # --- 1. Initialization ---
    log_file = os.path.join(PROJECT_ROOT, "run.log")
    setup_logger(log_file)
    set_seed(42)
    device = TRAIN_CONFIG['device']
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")

    # --- 2. Load and Preprocess Data ---
    logger.info("Starting to load the dataset...")
    dataset = RecDataset(
        file_path=DATA_CONFIG['file_path'],
        min_seq_len=DATA_CONFIG['min_seq_len']
    )
    
    # --- 3. Generate or Load Semantic IDs ---
    item_to_semantic_id, _ = get_semantic_ids(dataset, device)

    # --- 4. Build Trie for Constrained Decoding ---
    # Define special tokens. Codebook IDs range from 0 to codebook_size-1.
    # We will place the special tokens after the codebook IDs.
    special_tokens = {
        'PAD': 0, # Use 0 as the PAD token
        'BOS': RQ_VAE_CONFIG['codebook_size'],
        'EOS': RQ_VAE_CONFIG['codebook_size'] + 1,
        'UNK': RQ_VAE_CONFIG['codebook_size'] + 2,
    }
    # PAD is not a new token as it reuses an existing ID
    num_semantic_tokens = RQ_VAE_CONFIG['codebook_size'] + len(special_tokens) - 1 
    
    logger.info("Starting to build the Trie...")
    trie = build_trie_from_id_map(item_to_semantic_id, special_tokens)
    logger.info("Trie construction complete.")

    # --- 5. Split Dataset and Create DataLoaders ---
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=TRAIN_CONFIG['num_workers']
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=EVAL_CONFIG['batch_size'],
        collate_fn=collate_fn,
        num_workers=TRAIN_CONFIG['num_workers']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_CONFIG['batch_size'],
        collate_fn=collate_fn,
        num_workers=TRAIN_CONFIG['num_workers']
    )
    logger.info(f"Dataset split complete: Train={len(train_dataset)}, Valid={len(valid_dataset)}, Test={len(test_dataset)}")

    # --- 6. Initialize Model, Optimizer, and Loss Function ---
    model = DSRRec(
        num_items=dataset.item_count,
        num_semantic_tokens=num_semantic_tokens,
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        ffn_dim=MODEL_CONFIG['ffn_dim'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
        max_seq_len=SL_CONFIG['max_len_nc'] * 2 
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    # ignore_index=0 means that the loss for padded positions will not be calculated
    loss_fn = nn.CrossEntropyLoss(ignore_index=special_tokens['PAD'])
    logger.info("Model, optimizer, and loss function initialized.")


    # --- 7. Start Training ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        trie=trie,
        special_tokens=special_tokens,
        semantic_id_map=item_to_semantic_id
    )
    
    logger.info("Starting model training...")
    trainer.train(epochs=TRAIN_CONFIG['epochs'])


if __name__ == '__main__':
    main()