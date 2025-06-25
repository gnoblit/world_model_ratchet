import os
from src.configs.base_config import get_base_config
from src.training.trainer import Trainer

def main():
    """Main function to run the training process."""
    config = get_base_config()
    
    # Ensure experiment directory exists
    os.makedirs(config.experiment_dir, exist_ok=True)
    
    # Initialize the trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()