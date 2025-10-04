# coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class RQVAE(nn.Module):
    """
    RQ-VAE
    """
    def __init__(self, input_dim, output_dim, num_levels, codebook_size, embedding_dim, beta=0.25):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        
        # Multi level codebook
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, embedding_dim) for _ in range(num_levels)
        ])
        
        # The weight for the commitment loss, used to stabilize training.
        self.beta = beta
        self.num_levels = num_levels

    def quantize(self, x):
        """Performs multi level quantization on the input vectors."""
        residual = x
        quantized_indices = []
        
        for i in range(self.num_levels):
            codebook = self.codebooks[i]
            
            # Calculate the distance between input and all vectors in the codebook
            distances = torch.sum(residual**2, dim=1, keepdim=True) \
                      - 2 * torch.matmul(residual, codebook.weight.t()) \
                      + torch.sum(codebook.weight**2, dim=1, keepdim=True).t()

            # Find the index of the closest vector in the codebook
            indices = torch.argmin(distances, dim=1)
            quantized_indices.append(indices)
            
            # get the quantized vector
            quantized = codebook(indices)
            
            # update the residual
            residual = residual - quantized
            
        return torch.stack(quantized_indices, dim=1), residual

    def quantize_and_get_loss(self, x):
        """get the quantized vectors, indices, and the commitment loss."""
        residual = x
        quantized_vectors = []
        indices_list = []
        total_commitment_loss = 0.0

        for i in range(self.num_levels):
            codebook = self.codebooks[i]
            distances = torch.sum(residual**2, dim=1, keepdim=True) \
                      - 2 * torch.matmul(residual, codebook.weight.t()) \
                      + torch.sum(codebook.weight**2, dim=1, keepdim=True).t()

            indices = torch.argmin(distances, dim=1)
            indices_list.append(indices)
            
            quantized = codebook(indices)
            quantized_vectors.append(quantized)

            # calculate the commitment loss
            commitment_loss = F.mse_loss(residual.detach(), quantized)
            total_commitment_loss += commitment_loss
            
            # Straight-Through Estimator
            quantized_for_grad = residual + (quantized - residual).detach()
            residual = residual - quantized_for_grad
        
        final_quantized = torch.sum(torch.stack(quantized_vectors, dim=0), dim=0)
        return final_quantized, torch.stack(indices_list, dim=1), total_commitment_loss

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, indices, commitment_loss = self.quantize_and_get_loss(encoded)
        reconstructed = self.decoder(quantized)

        reconstruction_loss = F.mse_loss(reconstructed, x)
        
        # use beta weight commitment loss
        loss = reconstruction_loss + self.beta * commitment_loss
        
        return loss, reconstructed, indices

def train_rqvae(model, data_loader, optimizer, epochs, device):
    """train RQ-VAE"""
    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        for i, batch in enumerate(progress_bar):
            batch_tensor = batch[0].to(device)
            
            optimizer.zero_grad()
            loss, _, _ = model(batch_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(data_loader)
        logger.info(f"Epoch {epoch+1}, avg_loss: {avg_loss:.4f}")

