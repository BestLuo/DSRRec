
import torch

# --- Basic Configuration ---

PROJECT_ROOT = "mainpath"

# --- Dataset Configuration ---
DATA_CONFIG = {
    'file_path': 'datapath',
    'min_seq_len': 5  # 用Minimum user interaction sequence length; users with shorter sequences will be filtered.
}

# --- Text Encoder Qwen3-Embedding Configuration ---

QWEN3_EMBEDDING_CONFIG = {
    'model_path': "/root/soft/Rec/llm_models/BaseModel/models--Qwen--Qwen3-Embedding-4B/snapshots/408b81b7fab742073065d5b3661fa74c1b3ee0a1",
    'max_length': 512, # Maximum input length for text encoding.
    'output_dim': 1024 # Embedding dimension.
}

# --- RQ-VAE Configuration ---
# For generating semantic IDs
RQ_VAE_CONFIG = {
    'num_levels': 3,          # L, Number of quantization levels.
    'codebook_size': 256,     # N_k, Size of the codebook for each level.
    'embedding_dim': 32,      # d_k, Dimension of the codebook vectors for each level.
    'input_dim': QWEN3_EMBEDDING_CONFIG['output_dim'], # Input vector dimension
    'output_dim': QWEN3_EMBEDDING_CONFIG['output_dim'],# Output vector dimension
    'epochs': 20000,          # steps 
    'batch_size': 1024,
    'learning_rate': 0.4,
    'optimizer': 'Adagrad'    # Optimizer for RQ-VAE.
}

# --- DSR-Rec Model Configuration ---
MODEL_CONFIG = {
    'hidden_dim': 300,        # Hidden dimension of the Transformer layers.
    'num_layers': 4,          # Number of layers for the Encoder and Decoder.
    'num_heads': 6,           # Number of heads for the multi head attention mechanism.
    'ffn_dim': 1024,          # Dimension of the feed forward network's intermediate layer.
    'dropout_rate': 0.2,      #  dropout rate
    'semantic_id_length': RQ_VAE_CONFIG['num_levels'], # Length of the semantic ID.
    'codebook_size': RQ_VAE_CONFIG['codebook_size']    # codebook size
}

# --- CSMoE Configuration ---
CSMOE_CONFIG = {
    'num_experts': 8,         # Ne, Number of expert networks.
    'top_k': 2                # K, Number of experts to activate for each input.
}

# --- SL Configuration ---
SL_CONFIG = {
    'max_len_nc': 15,         # Nc, Maximum length parameter.
    'alpha': 1.5              # α, Sparsification hyperparameter, set to 1.5 in the paper for optimal balance.
}


# --- Training Configuration ---
TRAIN_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'epochs': 200,
    'batch_size': 128,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'num_workers': 18, # Number of processes used by DataLoader. 
    'eval_every_n_epochs': 5, # Perform evaluation every n epochs.
    'patience': 10 # Patience value for early stopping.
}

# --- Evaluation Configuration ---
EVAL_CONFIG = {
    'batch_size': 10000,
    'k_values': [5, 10] 
}
