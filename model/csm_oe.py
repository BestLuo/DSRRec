
import torch
import torch.nn as nn
import torch.nn.functional as F

class CSMoE(nn.Module):
    """
    Context-aware Sparse Mixture-of-Experts (CSMoE) layer.
    This layer replaces the standard Transformer decoder's feed-forward network (FFN).
    It uses the current decoding context to dynamically select a subset of experts for input processing.
    """
    def __init__(self, input_dim, ffn_dim, num_experts, top_k, dropout_rate):
        """
        Initializing the CSMoE Layer.

        Args:
            input_dim (int): The dimension of the input and output(d_model).
            ffn_dim (int): The dimension of the intermediate layer in each expert network.
            num_experts (int): The total number of expert networks.
            top_k (int): The number of experts to select for each forward pass.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Create list of expert networks
        # Each expert is a standard feed forward network.
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, ffn_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(ffn_dim, input_dim)
            ) for _ in range(num_experts)
        ])

        # Create the gating network.
        # It receives the context vector and outputs a score for each expert.
        self.gating_network = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        """
        Forward Pass of the CSMoE Layer.

        Args:
            x (torch.Tensor): The input to the FFN comes from the multi head attention sublayer.
            shape: (batch_size, seq_len, input_dim)

            context (torch.Tensor): The context vector, which is the output from the cross-attention sublayer. It fuses historical sequence information with currently generated information.
            shape: (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: Output of the CSMoE layer. Its shape is the same as the input x.
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Use the gating network to score each expert.
        # (B, S, D) -> (B, S, N_e)
        logits = self.gating_network(context)
        
        # 2. Select the top-k experts.
        # topk返回 (values, indices)
        # scores_topk: (B, S, K), indices_topk: (B, S, K)
        scores_topk, indices_topk = torch.topk(logits, self.top_k, dim=-1)
        
        # 3. Apply Softmax to the scores of the top-k experts to obtain the weights.
        # weights: (B, S, K)
        weights = F.softmax(scores_topk, dim=-1)

        # Initialize a zero tensor with the same shape as the input x to accumulate the expert outputs.
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, self.input_dim) # (B*S, D)
        flat_indices = indices_topk.view(-1) # (B*S*K)
        
        # Create a routing mask to indicate which token is processed by which expert.
        # (B*S, N_e)
        routing_mask = torch.zeros(batch_size * seq_len, self.num_experts, device=x.device, dtype=torch.bool)
        routing_mask.scatter_(1, indices_topk.view(-1, self.top_k), 1)

        # 4. Route the input to the selected experts and calculate the weighted output.
        for i in range(self.num_experts):
            # Find the indices of tokens to be processed by the current expert.
            expert_mask = routing_mask[:, i]
            
            if expert_mask.any():
                # Get the input for these tokens.
                expert_inputs = flat_x[expert_mask]
                
                # Process through the expert network.
                expert_outputs = self.experts[i](expert_inputs)
                
                # Get the weights corresponding to these tokens.
                # (num_active_tokens, K)
                weights_for_expert = weights.view(-1, self.top_k)[expert_mask]
                # (B*S*K) -> (B*S, K)
                indices_for_expert = indices_topk.view(-1, self.top_k)[expert_mask]
                
                # Find the position of the current expert in the top-k list to obtain the correct weights.
                # (num_active_tokens, 1)
                weight_indices = (indices_for_expert == i).nonzero(as_tuple=True)[1].unsqueeze(1)
                
                # (num_active_tokens, 1)
                correct_weights = torch.gather(weights_for_expert, 1, weight_indices)
                
                # Weighted output.
                weighted_output = expert_outputs * correct_weights
                
                # Place the weighted output into the correct positions within the final output tensor.
                final_output.view(-1, self.input_dim)[expert_mask] += weighted_output

        return final_output

