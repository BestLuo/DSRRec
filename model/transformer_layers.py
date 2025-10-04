# coding:utf-8
import torch
import torch.nn as nn
import math

from csm_oe import CSMoE
from config import MODEL_CONFIG, CSMOE_CONFIG

class MultiHeadAttention(nn.Module):
    """mutili head attention"""
    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.d_k = hidden_dim // num_heads
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transform and split into multiple heads.
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention scores.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum.
        context = torch.matmul(attention_weights, v)
        
        # Combine the multiple heads.
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Final linear transformation.
        output = self.out_linear(context)
        return output

class PositionwiseFeedForward(nn.Module):
    """Standard feed forward network."""
    def __init__(self, hidden_dim, ffn_dim, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class PositionalEncoding(nn.Module):
    """Positional Encoding"""
    def __init__(self, hidden_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    """The structure of a single Transformer Encoder Layer."""
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)
        self.ffn = PositionwiseFeedForward(hidden_dim, ffn_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        # self attention.
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        # feed forward network.
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

class DecoderLayer(nn.Module):
    """The Structure of a Single Transformer Decoder Layer with Integrated CSMoE."""
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)
        self.cross_attn = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)
        
        # Replacing the Standard FFN with CSMoE.
        self.csm_oe = CSMoE(
            input_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=CSMOE_CONFIG['num_experts'],
            top_k=CSMOE_CONFIG['top_k'],
            dropout_rate=dropout_rate
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, memory, src_mask, tgt_mask):
        # mask self attention.
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # cross attention.
        cross_attn_output = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # CSMoE layer.
        # The context for the CSMoE is a fusion of global collaborative information (from the encoder) and local semantic information (from the decoder).
        # cross_attn_output is the result of this fusion and is therefore used as the context input.
        csm_oe_output = self.csm_oe(x, context=cross_attn_output)
        x = self.norm3(x + self.dropout3(csm_oe_output))
        
        return x
