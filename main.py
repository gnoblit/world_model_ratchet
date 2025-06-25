import os
from configs.base_config import get_base_config
# Import the new manager
from training.iterated_learning import IteratedLearningManager

def main():
    """Main function to run the training process."""
    config = get_base_config()
    
    # Ensure experiment directory exists
    os.makedirs(config.experiment_dir, exist_ok=True)
    
    # --- CHOOSE YOUR EXPERIMENT ---
    # To run the IL experiment, use the manager.
    # To run the baseline experiment, you would instantiate and call the Trainer directly.
    
    print("Starting Iterated Learning Experiment...")
    manager = IteratedLearningManager(config)
    manager.run_il_loop()

    # --- To run the baseline for comparison, you would comment out the above and use this:
    # from training.trainer import Trainer
    # print("Starting Baseline Experiment...")
    # trainer = Trainer(config)
    # trainer.train_for_steps(config.training.total_train_steps) # A new total steps param might be needed

if __name__ == "__main__":
    main()