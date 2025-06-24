import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedCodebook(nn.Module):
    """
    A learnable codebook of discrete latent codes, as described in the IL-CLIP paper.
    Takes continuous features and maps them to a sparse combination of its codes.
    """

    def __init__(self, num_codes: int, code_dim: int):
        """
        Initializes the SharedCodebook.
        
        Args:
            num_codes (int): The number of discrete codes in the codesbook (C in the paper).
            code_dim (int): The dimensionality of each code vector.
        """

        super().__init__()
        # Codebook is a standard embedding layer
        self.embedding = nn.Embedding(num_codes, code_dim)
        self.num_codes = num_codes
        self.code_dim = code_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Takes continuous input features and maps them to a sparse combination of codes.
        
        Args:
            features (torch.Tensor): A batch of continuous feature vectors.
                                                Shape: (batch_size, feature_dim).
                                                Note: feature_dim must equal code_dim.
                                                
        Returns:
            torch.Tensor: The final representation, which is the weighted sum of codes.
                                Shape: (batch_size, code_dim)."""
        
        if features.shape[-1] != self.code_dim:
            raise ValueError(
                f"Input feature dimension ({features.shape[-1]} must match "
                f"codebook dimension ({self.code_dim})"
            )
        
        # Get all code vectors from the embedding table
        # Shape: (num_codes, code_dim)
        codes = self.embedding.weight
        
        # Normalize features and codes for cosine similarity calculation
        features_norm = F.normalize(features, p=2, dim=-1)       # Shape: (batch_size, code_dim)
        codes_norm = F.normalize(codes, p=2, dim=-1)              # Shape: (num_codes, code_dim)

        # Calculate cosine similarity between each feature and all codes
        # Matrix multiplication: (B, D) @ (D, C) -> (B, C)
        similarity_scores = torch.matmul(features_norm, codes_norm.T)
        # Shape: (batch_size, num_codes)

        # Get weights
        weights = F.softmax(similarity_scores, dim=-1)

        # Calculate final representation as weighted sum of the original non_normalized codes
        # Matmul: (B, C) @ (C, D) -> (B, D)
        final_representation = torch.matmul(weights, codes)
        # Shape: (batch_size, code_dim)

        return final_representation