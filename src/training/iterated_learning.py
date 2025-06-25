import torch
import torch.optim as optim
from tqdm import tqdm

from configs.base_config import MainConfig
from training.trainer import Trainer
from models.actor_critic import ActorCritic

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

    def spawn_new_student(self):
        """
        Re-initializes the student (ActorCritic) and its optimizer.
        This simulates the "generational shift".
        """
        # Get the necessary dimensions from the config
        state_dim = self.cfg.perception.code_dim
        
        # Create a new instance of the ActorCritic model
        new_student = ActorCritic(state_dim=state_dim, cfg=self.cfg.action).to(self.trainer.device)
        
        # Replace the old student in the trainer with the new one
        self.trainer.actor_critic = new_student
        
        # Create a new optimizer for the new student
        new_optimizer = optim.Adam(
            self.trainer.actor_critic.parameters(), 
            lr=self.cfg.training.action_model_lr
        )
        
        # Replace the old optimizer in the trainer
        self.trainer.action_optimizer = new_optimizer
        
        print("Successfully spawned a new student agent and its optimizer.")

    def run_il_loop(self):
        """The main loop for conducting the iterated learning experiment."""
        num_generations = self.cfg.il.num_generations
        
        print(f"Starting Iterated Learning for {num_generations} generations.")

        # --- Initial Warmup (Generation 0) ---
        print("\n--- Running Generation 0 (Warmup) ---")
        self.trainer.train_for_steps(self.cfg.il.warmup_steps, teacher_is_frozen=False)
        
        for generation in range(1, num_generations + 1):
            print(f"\n--- Starting Generation {generation}/{num_generations} ---")

            # --- 1. Generational Shift: Spawn a new student ---
            self.spawn_new_student()

            # --- 2. Distillation Phase ---
            print(f"Starting distillation for {self.cfg.il.distill_steps} steps...")
            # Run training with the teacher (perception) frozen
            self.trainer.train_for_steps(self.cfg.il.distill_steps, teacher_is_frozen=True)

            # --- 3. Interaction Phase ---
            print(f"Starting interaction for {self.cfg.il.interact_steps} steps...")
            # Run training with all models unfrozen
            self.trainer.train_for_steps(self.cfg.il.interact_steps, teacher_is_frozen=False)

        print("\nIterated Learning finished.")
        # Final cleanup
        self.trainer.close()