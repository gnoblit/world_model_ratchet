
import os
import argparse
import time
import uuid 

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
        # Set learning_starts to a value smaller than the warmup/student steps
        # to ensure training actually happens during these short phases.
        # The original value of 500 prevented any updates in the first 400 steps.
        config.training.learning_starts = 100
        # For Baseline
        config.training.total_train_steps = 5_000 
        # For IL
        config.il.num_generations = 2
        config.il.warmup_env_steps = 200
        config.il.student_env_steps = 200
        config.il.teacher_grad_updates = 100

    # --- Set run name based on experiment type ---

    # Get the current date as a string
    current_time = time.time()
    datestamp = time.strftime('%Y-%m-%d_%H-%M', time.localtime(current_time)).replace("-", "_")
    # Generate a short, unique hex ID (e.g., 'a1b2c3')
    unique_id = uuid.uuid4().hex[:6]
    # Store
    config.run_timestamp = current_time
    config.run_uuid = unique_id

    # Determine the base name for the run
    if args.run_name:
        base_name = args.run_name
    else:
        mode = "il" if args.experiment == "il" else "baseline"
        mini_str = "_mini" if args.mini else ""
        base_name = f"{mode}{mini_str}"
    
    # Combine the datestamp and base name for the final run name
    config.run_name = f"{datestamp}_{base_name}_{unique_id}"

    # --- Setup Logging and Config Saving ---
    log_dir = None
    if config.run_name:
        log_dir = os.path.join(config.experiment_dir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        save_config(config, log_dir)
    
    # --- Launch the selected experiment ---
    if args.experiment == "baseline":
        print(f"--- Starting Baseline Experiment: {config.run_name} ---")
        trainer = Trainer(config)
        trainer.train_for_steps(
            num_env_steps=config.training.total_train_steps, 
            teacher_is_frozen=False
        )
        trainer.close()
    
    elif args.experiment == "il":
        print(f"--- Starting Iterated Learning Experiment: {config.run_name} ---")
        manager = IteratedLearningManager(config)
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