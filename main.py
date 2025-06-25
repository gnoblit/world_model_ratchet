import os
from configs.base_config import get_base_config
from utils.config_utils import save_config
from training.trainer import Trainer
# from training.iterated_learning import IteratedLearningManager # Don't need this for baseline

def main():
    """Main function to run the training process."""
    config = get_base_config()
    
    # This part is fine. It will create experiments/baseline_v1/
    log_dir = os.path.join(config.experiment_dir, config.run_name)
    os.makedirs(log_dir, exist_ok=True)
    save_config(config, log_dir)
    
    # --- Run the Baseline Experiment ---
    print("--- Starting Baseline Experiment ---")
    
    # Your __init__ creates the logger internally, so we don't need to pass the dir
    trainer = Trainer(config) 
    
    # We call the main training method once for the total duration
    trainer.train_for_steps(
        num_steps=config.training.total_train_steps, 
        teacher_is_frozen=False
    )
    
    # Clean up and save models
    trainer.close()

if __name__ == "__main__":
    main()