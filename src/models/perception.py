import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VisionEncoder(nn.Module):
    """
    A CNN that takes an image observation and encodes it into a flat feature vector.
    Uses a pre-defined architecture from torchvision (e.g., ResNet-18) and adapts it.
    """
    def __init__(self, feature_dim: int, pretrained: bool=False):
        """
        Initializes the VisionEncoder.
        
        Args:
            feature_dim (int): The desired dimensionality of the output feature vector.
                                        This must match the code_dim of the SharedCodebook.
            pretrained (bool): Whether to use pre-trained weights from ImageNet.
                                        Should be False for our task, as we learn from scratch.
        """

        super().__init__()

        # 1. Load model from torchvision
        # Use weights=None for random initialization
        resnet = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)

        # 2. Isolate feature extraction layers
        # Take all but final classification layer, which we drop
        modules = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)

        # 3. Create a new head to project the features to our desired dimension
        in_features = resnet.fc.in_features
        self.head = nn.Linear(in_features, feature_dim)

        self.feature_dim = feature_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Processes a batch of image observations.
        
        Args:
            obs (torch.Tensor): A batch of image tensors.
                                          Shape: (batch_size, channels, heigh, width)

        Returns:
            torch.Tensor: A batch of flat feature vectors.
                                 Shape: (batch_size, feature_dim)                                          
        """
        # Pass through feature extractor
        features = self.feature_extractor(obs)

        # Flatten to be 1D vector per image
        features_flat = torch.flatten(features, 1)

        # Project to the final feature dimension
        output_features = self.head(features_flat)

        return output_features


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
    
    class PerceptionAgent(nn.Module):
        """
        The complete "Teacher" module. It combines the VisionEncoder and the
        SharedCodebook to process raw observations into a final, structured
        representation.
        """
        def __init__(self, cfg):
            """
            Initializes the PerceptionAgent.

            We will pass a configureation object to keep the parameters organized.
            
            Args:
                cfg (ModelConfig): A configuration object containing the model hyperparameters.
            """
            super().__init__()

            # We will expect the config to have:
            # cfg.perception.feature_dim
            # cfg.perception.num_codes
            # cfg.perception.code_dim

            # Feature dimension of the encoder's output must
            # match the code dim of the codebook's input/output

            if cfg.feature_dim != cfg. code_dim:
                raise ValueError("""
                                 VisionEncoder's feature dim must match SharedCodebook's code_dim.
                                 Check config.
                                 """)
            self.encoder = VisionEncoder(
                feature_dim=cfg.feature_dim,
            )
            self.codebook = SharedCodebook(
                num_codes=cfg.num_codes,
                code_dim=cfg.code_dim,
            )

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            """
            The full forward pass from pixels to structured representation.
            
            Args:
                obs (torch.Tensor): A batch of image observations.
                                             Shape: (batch_size, channels, height, width)
                
            Returns:
                torch.Tensor: The final representation `z_t` from the codebook. 
                                    Shape: (batch_size, code_dim)
            """

            # 1. Encode the raw pixes into a continuous feature vector
            features = self.encoder(obs)
            # 2. Map the continuous features to a sparse combination of discrete codes
            representation = self.codebook(features)

            return representation