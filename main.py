import os
from configs.base_config import get_base_config
from training.iterated_learning import IteratedLearningManager
from utils.config_utils import save_config

def main():
    config = get_base_config()
    
    # Only create directories and save config if a run name is specified
    if config.run_name:
        log_dir = os.path.join(config.experiment_dir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        # Save the config for this run
        save_config(config, log_dir) 
    else:
        log_dir = None
    
    print("Starting Iterated Learning Experiment...")
    manager = IteratedLearningManager(config, log_dir)
    manager.run_il_loop()

if __name__ == "__main__":
    main()