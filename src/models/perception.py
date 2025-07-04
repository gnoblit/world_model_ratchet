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
            torch.Tensor: A batch of flat feature vectors. Shape: (batch_size, feature_dim)
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

    def forward(self, features: torch.Tensor, compute_losses: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Takes continuous input features and maps them to a sparse combination of codes.
        
        Args:
            features (torch.Tensor): A batch of continuous feature vectors.
                                                Shape: (batch_size, feature_dim).
                                                Note: feature_dim must equal code_dim.
            compute_losses (bool): If False, skips loss calculations for performance.
                                                
        Returns:
            A tuple (final_representation, codebook_loss, commitment_loss, code_entropy):
                - final_representation (torch.Tensor): The final representation `z_t`. Uses a straight-through estimator to allow gradients to flow back to the encoder.
                - codebook_loss (torch.Tensor): Loss to move codebook vectors towards encoder outputs.
                - commitment_loss (torch.Tensor): Loss to move encoder outputs towards codebook vectors.
                - code_entropy (torch.Tensor): Entropy of code usage in the batch, used as a regularization loss to encourage using more of the codebook.
        """

        # Input validation

        if features.shape[-1] != self.code_dim:
            raise ValueError(
                f"Input feature dimension ({features.shape[-1]} must match "
                f"codebook dimension ({self.code_dim})"
            )
        
        codes = self.embedding.weight
        batch_size = features.shape[0]
        
        # --- Hard Vector Quantization ---
        # Calculate the L2 squared distance between each feature and all codes
        # (a-b)^2 = a^2 - 2ab + b^2
        distances = (
            torch.sum(features**2, dim=1, keepdim=True)
            + torch.sum(codes**2, dim=1)
            - 2 * torch.matmul(features, codes.t())
        ) # Shape: (batch_size, num_codes)

        # Find the index of the closest codebook vector for each feature vector
        closest_code_indices = torch.argmin(distances, dim=1) # Shape: (batch_size,)

        # The quantized representation is the selected codebook vector
        quantized_features = self.embedding(closest_code_indices)
        
        codebook_loss, commitment_loss, code_entropy = None, None, None
        if compute_losses:
            # --- Calculate Loss Terms ---
            
            # 1. Codebook Loss: Encourages the codebook vectors (e_k) to move towards
            # the encoder's outputs (features). We stop the gradient on the features
            # so we only update the codebook.
            codebook_loss = F.mse_loss(quantized_features, features.detach())

            # 2. Commitment Loss: Encourages the encoder's output (features) to be
            # "committed" to the chosen code vector.
            # We stop the gradient on the quantized features so the encoder is pulled
            # towards the codes, not the other way around.
            commitment_loss = F.mse_loss(features, quantized_features.detach())
            
            
            # 3. Differentiable Codebook Entropy Loss: To encourage diverse code usage,
            # we compute the entropy of a "soft" distribution over the codes. This is
            # differentiable w.r.t. the input features, unlike an entropy based on argmin.
            soft_probs = F.softmax(-distances, dim=1) # Smaller distances get higher probability

            # Calculate the entropy for each sample's probability distribution over the codes.
            # Then, average these entropies across the batch. This is generally more stable
            # than calculating the entropy of the batch's average distribution.
            per_sample_entropy = -torch.sum(soft_probs * torch.log(soft_probs + 1e-9), dim=1)
            code_entropy = per_sample_entropy.mean()
        
        # We subtract this entropy term from the total loss to maximize it.

        # We also need to allow the gradient to flow back from the final_representation
        # to the encoder. The commitment loss helps the encoder, but the main task
        # gradient still needs to pass through. This is a common trick in VQ-VAEs.
        # The gradient from `final_representation` will flow back to `features`.
        final_representation = features + (quantized_features - features).detach()

        return final_representation, codebook_loss, commitment_loss, code_entropy
    
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
            cfg (PerceptionConfig): A configuration object containing the model hyperparameters.
        """
        super().__init__()

        if cfg.feature_dim != cfg.code_dim:
            raise ValueError(
                "VisionEncoder's feature dim must match SharedCodebook's code_dim. "
                f"Got feature_dim={cfg.feature_dim} and code_dim={cfg.code_dim}."
            )

        self.encoder = VisionEncoder(
            feature_dim=cfg.feature_dim,
            pretrained=cfg.pretrained
        )
        self.codebook = SharedCodebook(
            num_codes=cfg.num_codes,
            code_dim=cfg.code_dim,
        )

    def forward(self, obs: torch.Tensor, compute_losses: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The full forward pass from pixels to structured representation.
        
        Args:
            obs (torch.Tensor): A batch of image observations.
                                            Shape: (batch_size, channels, height, width)
            compute_losses (bool): If False, skips VQ loss calculations for performance.
            
        Returns:
            A tuple (representation, codebook_loss, commitment_loss, code_entropy):
                representation (torch.Tensor): The final representation `z_t`.
                codebook_loss (torch.Tensor): The loss from the codebook.
                commitment_loss (torch.Tensor): The loss from the codebook.
                code_entropy (torch.Tensor): The entropy of the codebook usage.
        """

        # 1. Encode the raw pixes into a continuous feature vector
        features = self.encoder(obs)
        # 2. Map the continuous features to a sparse combination of discrete codes
        representation, codebook_loss, commitment_loss, code_entropy = self.codebook(features, compute_losses=compute_losses)

        return representation, codebook_loss, commitment_loss, code_entropy