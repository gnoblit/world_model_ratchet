from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class EnvConfig:
    """Configuration for the environmnet."""
    env_name: str = "Crafter"
    seed: int = 666
    image_size: Tuple[int, int] = (64, 64)

@dataclass
class MainConfig:
    """Main configuration for the project."""
    project_name: str = "world_model_ratchet"
    experiment_dir: str = "experiments"

    env: EnvConfig = field(default_factory=EnvConfig)

def get_base_config():
    return MainConfig()