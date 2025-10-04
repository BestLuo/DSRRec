# coding:utf-8
import torch
import torch.nn as nn
import math
from transformer_layers import PositionalEncoding, EncoderLayer, DecoderLayer
from config import MODEL_CONFIG

class DSRRec(nn.Module):
    """
    The Overall Architecture of the DSR-Rec Model.
    This is a Transformer-based Encoder-Decoder model specifically designed for sequential recommendation.
    The Encoder is responsible for encoding the user's historical interaction sequence to generate a collaborative information representation.
    The Decoder is an autoregressive model that, based on the collaborative information and the partially generated semantic information, sequentially predicts the next token of the target item's semantic ID.
    """
    def __init__(self, num_items, num_semantic_tokens, hidden_dim, num_layers, num_heads, ffn_dim, dropout_rate, max_seq_len):
        """
        Initialize the DSR-Rec model.
        Args:
            num_items (int): The total number of unique items in the dataset (used for historical sequence embedding).
            num_semantic_tokens (int): The vocabulary size for semantic IDs(codebook_size + number of special tokens).
            hidden_dim (int): The dimension of the model's hidden layers(d_model).
            num_layers (int): The number of layers in the Encoder and Decoder.
            num_heads (int): The number of heads for the multi head attention mechanism.
            ffn_dim (int): The dimension of the intermediate layer in the FFN/CSMoE.
            dropout_rate (float): Dropout rate.
            max_seq_len (int): The maximum sequence length.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Item Embedding Layer (for Encoder input).
        # padding_idx=0 indicates that 0 is the padding ID; its embedding vector will always be zero and will not participate in gradient updates.
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        
        # Semantic ID Token Embedding Layer (for Decoder input).
        self.semantic_token_embedding = nn.Embedding(num_semantic_tokens, hidden_dim, padding_idx=0)
        
        # Positional Encoding.
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Encoder
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(hidden_dim, num_heads, ffn_dim, dropout_rate) for _ in range(num_layers)]
        )
        
        # Decoder
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(hidden_dim, num_heads, ffn_dim, dropout_rate) for _ in range(num_layers)]
        )
        
        # Final Output Layer.
        self.output_layer = nn.Linear(hidden_dim, num_semantic_tokens)
        
        self.dropout = nn.Dropout(dropout_rate)

    def _create_padding_mask(self, seq):
        # seq: (batch_size, seq_len)
        # return: (batch_size, 1, 1, seq_len)
        return (seq != 0).unsqueeze(1).unsqueeze(2)

    def _create_subsequent_mask(self, seq):
        # seq: (batch_size, seq_len)
        seq_len = seq.size(1)
        # (1, seq_len, seq_len)
        subsequent_mask = torch.triu(torch.ones((seq_len, seq_len), device=seq.device), diagonal=1).bool()
        return subsequent_mask == 0 # 0 represents the positions to be masked.
    
    def encode(self, history_seq):
        """
        Encoder Forward Pass.
        Args:
            history_seq (torch.Tensor): The user's historical interaction sequence. shape: (batch_size, history_len).
            
        Returns:
            torch.Tensor: Encoded historical sequence representation.
            shape:(batch_size, history_len, hidden_dim).
        """
        src_mask = self._create_padding_mask(history_seq) # (B, 1, 1, S)
        
        # Item embedding and positional encoding.
        src = self.item_embedding(history_seq) * math.sqrt(self.hidden_dim)
        src = self.positional_encoding(src)
        src = self.dropout(src)
        
        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)
            
        return memory, src_mask

    def decode(self, target_seq, memory, src_mask):
        """
        Decoder Forward Pass.
        Args:
            target_seq (torch.Tensor): Target Semantic ID Sequence (Decoder input).
            shape:(batch_size, target_len).

            memory (torch.Tensor): Output of the Encoder.
            shape:(batch_size, history_len, hidden_dim).

            src_mask (torch.Tensor): Padding mask for the source historical sequence.
            
        Returns:
            torch.Tensor: Logits from the Decoder output.
            shape: (batch_size, target_len, num_semantic_tokens).
        """
        tgt_padding_mask = self._create_padding_mask(target_seq) # (B, 1, 1, T)
        tgt_subsequent_mask = self._create_subsequent_mask(target_seq) # (T, T)
        tgt_mask = tgt_padding_mask & tgt_subsequent_mask # (B, 1, T, T)
        
        # Target token embedding and positional encoding.
        tgt = self.semantic_token_embedding(target_seq) * math.sqrt(self.hidden_dim)
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)
        
        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, memory, src_mask, tgt_mask)
            
        # The final linear layer outputs the logits.
        logits = self.output_layer(output)
        return logits

    def forward(self, history_seq, target_seq):
        """
        The full forward pass.
        
        Args:
            history_seq (torch.Tensor): Historical sequence. 
            shape: (batch_size, history_len).

            target_seq (torch.Tensor): Target semantic ID sequence. 
            shape: (batch_size, target_len).
            
        Returns:
            torch.Tensor: Predicted logits. 
            shape: (batch_size, target_len, num_semantic_tokens).
        """
        memory, src_mask = self.encode(history_seq)
        logits = self.decode(target_seq, memory, src_mask)
        return logits

    def predict(self, history_seq, max_len, trie, bos_token_id, eos_token_id):
        """
        Args:
            history_seq (torch.Tensor): The historical sequence for a single user.
            shape: (1, history_len).

            max_len (int): The maximum length of the semantic ID to generate.
            trie (Trie): A Trie used for constrained decoding.
            bos_token_id (int): The BOS token ID.
            eos_token_id (int): The EOS token ID.
        
        Returns:
            list: The generated top-1 semantic ID sequence (without BOS/EOS tokens).
        """
        self.eval()
        with torch.no_grad():
            memory, src_mask = self.encode(history_seq)
            
            # Initialize the decoder input, starting with the BOS token.
            generated_seq = torch.tensor([[bos_token_id]], device=history_seq.device)
            
            for _ in range(max_len):
                # Decode the current sequence.
                logits = self.decode(generated_seq, memory, src_mask)
                
                # We only need the prediction from the last time step.
                next_token_logits = logits[:, -1, :]
                
                # Trie-constrained decoding.
                current_prefix = generated_seq.squeeze(0).tolist()[1:] # Remove the BOS token.
                valid_next_tokens = trie.get_next_valid_tokens(current_prefix)
                
                # If there is no valid next token (e.g., a leaf node is reached), force the generation to end.
                if not valid_next_tokens:
                    next_token_id = eos_token_id
                else:
                    # Create a mask to retain only the logits for valid tokens.
                    valid_mask = torch.full_like(next_token_logits, -float('inf'))
                    valid_indices = torch.tensor(list(valid_next_tokens), device=valid_mask.device)
                    valid_mask.scatter_(1, valid_indices.unsqueeze(0), 0.0)
                    
                    
                    next_token_logits += valid_mask
                    
                    next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                
                # If the EOS token is predicted, stop the generation process.
                if next_token_id == eos_token_id:
                    break
                
                # Add the newly generated token to the sequence to be used as input for the next time step.
                generated_seq = torch.cat([generated_seq, torch.tensor([[next_token_id]], device=history_seq.device)], dim=1)

        # Return the generated sequence (with the BOS token removed).
        return generated_seq.squeeze(0).tolist()[1:]



