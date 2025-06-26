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
        num_actions = self.cfg.action.num_actions
        
        # Create a new instance of the ActorCritic model
        new_student = ActorCritic(state_dim=state_dim, num_actions=num_actions, 
                                  cfg=self.cfg.action).to(self.trainer.device)

        # --- Compile the new student if torch.compile is enabled ---
        # This is crucial to maintain performance consistency across generations.
        if self.cfg.training.use_torch_compile and self.trainer.device == 'cuda':
            print("Compiling new student model...")
            new_student = torch.compile(new_student, mode="reduce-overhead")
        
        # Replace the old student in the trainer with the new one
        self.trainer.actor_critic = new_student
        
        # Create a new optimizer for the new student
        new_optimizer = optim.Adam(
            new_student.parameters(), 
            lr=self.cfg.training.action_model_lr
        )
        
        # Replace the old optimizer in the trainer
        self.trainer.action_optimizer = new_optimizer
        
        print("Successfully spawned a new student agent and its optimizer.")

    def run_il_loop(self):
        """The main loop for conducting the iterated learning experiment."""
        num_generations = self.cfg.il.num_generations
        
        print(f"Starting Iterated Learning for {num_generations} generations.")

        # --- Clear Buffer Before Starting ---
        # Ensure a clean slate for the entire experiment, preventing data leakage
        # if the manager is ever re-used.
        print("Clearing replay buffer before new experiment...")
        self.trainer.replay_buffer.clear()

        # --- Initial Warmup (Generation 0) ---
        print("\n--- Running Generation 0 (Warmup) ---")
        self.trainer.train_for_steps(self.cfg.il.warmup_env_steps, teacher_is_frozen=False)
        
        for generation in range(1, num_generations + 1):
            print(f"\n--- Starting Generation {generation}/{num_generations} ---")

            # --- 1. Generational Shift: Spawn a new student ---
            self.spawn_new_student()

            # --- 2. Clear Buffer for New Generation ---
            # This is critical. We must clear the buffer to ensure the teacher
            # is refined ONLY on the data from its direct student.
            print("Clearing replay buffer for the new generation...")
            self.trainer.replay_buffer.clear()

            # --- 3. Student Training Phase ---
            print(f"Starting student training for {self.cfg.il.student_env_steps} env steps...")
            # The student trains by interacting with the env, but the teacher is frozen.
            self.trainer.train_for_steps(self.cfg.il.student_env_steps, teacher_is_frozen=True)

            # --- 4. Teacher Refinement Phase ---
            print("Starting teacher refinement phase...")            
            # The replay buffer now contains experience collected *only* by the newly
            # trained student. We use this data to refine the teacher.
            print(f"Refining teacher for {self.cfg.il.teacher_grad_updates} grad updates using existing buffer data...")
            self.trainer.train_from_buffer(num_updates=self.cfg.il.teacher_grad_updates)

            # --- 5. Save Intermediate Models ---
            # Save the state of the models at the end of each generation for fault tolerance
            # and for analyzing the progression of the teacher model.
            print(f"Saving models for generation {generation}...")
            self.trainer.save_models(suffix=f"_gen_{generation}")

        print("\nIterated Learning finished.")
        # Final cleanup
        self.trainer.close()