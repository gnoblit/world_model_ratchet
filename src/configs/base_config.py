from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class EnvConfig:
    """Configuration for the environmnet."""
    env_name: str = "Crafter"
    seed: int = 666
    image_size: Tuple[int, int] = (64, 64)

@dataclass
class PerceptionConfig:
    """Configuration for the Perception Agent."""
    # The dimension of the feature vector produced by the VisionEncoder
    feature_dim: int = 256
    # The number of discrete codes in our "vocabulary"
    num_codes: int = 512
    # The dimension of each code in the codebook. Must match feature_dim.
    code_dim: int = 256

@dataclass
class ActionConfig:
    """Configuration for the Action Agent."""
    num_actions: int = 17 # For Crafter
    hidden_dim: int = 256

@dataclass
class WorldModelConfig:
    """Configuration for the World Model."""
    hidden_dim: int = 512
    # The dimension for the learnable action embeddings
    action_embedding_dim: int = 32
    
@dataclass
class MainConfig:
    """Main configuration for the project."""
    project_name: str = "world_model_ratchet"
    experiment_dir: str = "experiments"

    env: EnvConfig = field(default_factory=EnvConfig)

    # Perception 
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    # ActionAgent
    action: ActionConfig = field(default_factory=ActionConfig)
    # WorldModel
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)

def get_base_config():
    return MainConfig()