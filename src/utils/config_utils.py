import os
from omegaconf import OmegaConf

def save_config(cfg, save_dir: str):
    """Saves an OmegaConf config object to a YAML file."""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "config.yaml")
    with open(file_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    print(f"Configuration saved to {file_path}")