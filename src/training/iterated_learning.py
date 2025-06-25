import torch
from tqdm import tqdm

from configs.base_config import MainConfig
from training.trainer import Trainer
from models.actor_critic import ActorCritic # Have to reinit student

class IteratedLearningManager:
    """
    Manages the Iterated Learning (IL) cycle, acting as a wrapper around the Trainer.
    It orchestrates the process of spawning new "student" agents and having them
    learn from a continuing "teacher" agent.
    """
    def __init__(self, cfg: MainConfig):
        self.cfg = cfg
        # The manager creates and owns the single Trainer instance
        self.trainer = Trainer(cfg)
        print("Iterated Learning Manager initialized.")

    def run_il_loop(self):
        """The main loop for conducting the iterated learning experiment."""
        num_generations = self.cfg.il.num_generations
        
        print(f"Starting Iterated Learning for {num_generations} generations.")

        # --- Initial Warmup (Generation 0) ---
        print("--- Running Generation 0 (Warmup) ---")
        # The trainer's `train` method needs to be adapted to run for a
        # specific number of steps, not just a total. We'll refactor it.
        # self.trainer.train_for_steps(self.cfg.il.warmup_steps)
        
        for generation in range(1, num_generations + 1):
            print(f"\n--- Starting Generation {generation}/{num_generations} ---")

            # --- 1. Generational Shift: Spawn a new student ---
            print("Spawning new student (ActionAgent)...")
            # TODO: Implement the logic to reset the student agent
            # self.spawn_new_student()

            # --- 2. Distillation Phase ---
            print(f"Starting distillation for {self.cfg.il.distill_steps} steps...")
            # TODO: Run training with the teacher (perception) frozen
            # self.trainer.train_for_steps(self.cfg.il.distill_steps, teacher_is_frozen=True)

            # --- 3. Interaction Phase ---
            print(f"Starting interaction for {self.cfg.il.interact_steps} steps...")
            # TODO: Run training with all models unfrozen
            # self.trainer.train_for_steps(self.cfg.il.interact_steps, teacher_is_frozen=False)

        print("\nIterated Learning finished.")