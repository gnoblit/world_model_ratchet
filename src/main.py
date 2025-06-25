
import os
import argparse
import time

# Corrected imports for a file inside the `src` package
from configs.base_config import get_base_config
from utils.config_utils import save_config
from training.trainer import Trainer
from training.iterated_learning import IteratedLearningManager

def main(args):
    """Main function to configure and run the training process."""
    config = get_base_config()

    # --- Override config for mini-run if specified ---
    if args.mini:
        print("--- RUNNING IN MINI MODE ---")
        config.training.learning_starts = 1000
        # For Baseline
        config.training.total_train_steps = 25_000 
        # For IL
        config.il.num_generations = 2
        config.il.warmup_steps = 5_000
        config.il.distill_steps = 5_000
        config.il.interact_steps = 5_000

    # --- Set run name based on experiment type ---
    if args.run_name:
        config.run_name = args.run_name
    else:
        # Create a default name if not provided
        mode = "il" if args.experiment == "il" else "baseline"
        mini_str = "_mini" if args.mini else ""
        config.run_name = f"{mode}{mini_str}_{int(time.time())}"

    # --- Setup Logging and Config Saving ---
    log_dir = None
    if config.run_name:
        log_dir = os.path.join(config.experiment_dir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        save_config(config, log_dir)
    
    # --- Launch the selected experiment ---
    if args.experiment == "baseline":
        print(f"--- Starting Baseline Experiment: {config.run_name} ---")
        trainer = Trainer(config, logger_dir=log_dir)
        trainer.train_for_steps(
            num_steps=config.training.total_train_steps, 
            teacher_is_frozen=False
        )
        trainer.close()
    
    elif args.experiment == "il":
        print(f"--- Starting Iterated Learning Experiment: {config.run_name} ---")
        manager = IteratedLearningManager(config, logger_dir=log_dir)
        manager.run_il_loop()

    else:
        raise ValueError(f"Unknown experiment type: {args.experiment}")

if __name__ == "__main__":
    # We need to import time for the default run_name
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="il", 
        choices=["baseline", "il"],
        help="The type of experiment to run."
    )
    parser.add_argument(
        "--run_name", 
        type=str, 
        default=None,
        help="A specific name for the experiment run. Overrides default naming."
    )
    parser.add_argument(
        "--mini", 
        action="store_true", # This makes it a flag, e.g., `python main.py --mini`
        help="If set, runs a short version of the experiment for debugging."
    )
    args = parser.parse_args()
    
    main(args)