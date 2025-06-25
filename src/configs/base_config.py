from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch

@dataclass
class EnvConfig:
    """Configuration for the environmnet."""
    env_name: str = "Crafter"
    seed: int = 666
    image_size: Tuple[int, int] = (64, 64)

@dataclass
class PerceptionConfig:
    """Configuration for the Perception Agent."""
    # The dimension of the feature vector produced by the VisionEncoder
    feature_dim: int = 256
    # The number of discrete codes in our "vocabulary"
    num_codes: int = 512
    # The dimension of each code in the codebook. Must match feature_dim.
    code_dim: int = 256

@dataclass
class ActionConfig:
    """Configuration for the Action Agent."""
    num_actions: int = 17 # For Crafter
    hidden_dim: int = 256

@dataclass
class WorldModelConfig:
    """Configuration for the World Model."""
    hidden_dim: int = 512
    # The dimension for the learnable action embeddings
    action_embedding_dim: int = 32

@dataclass
class ReplayBufferConfig:
    """Configuration for the Replay Buffer."""
    # Maximum number of *episodes* to store in the buffer.
    capacity: int = 10_000
    # The length of the sequences (trajectories) to sample
    sequence_length: int = 50

@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    # Move device selection into the config for better control
    # default_factory allows us to run code to determine the default value
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Learning rates for the optimizers
    world_model_lr: float = 1e-4
    action_model_lr: float = 3e-4
    
    # Total number of environment steps to train for
    total_train_steps: int = 1_000_000
    
    # How many steps to collect before starting to train the models
    learning_starts: int = 5000

    # Perform a model update every N environment steps.
    update_every_steps: int = 4 # A common value
    
    # The batch size for sampling from the replay buffer
    batch_size: int = 32

    # --- A2C specific hyperparameters ---
    # Discount factor for future rewards
    gamma: float = 0.99
    # Coefficient for the critic's value loss
    critic_loss_coef: float = 0.5
    # Coefficient for the entropy bonus to encourage exploration
    entropy_coef: float = 0.01
    # Gradient clipping
    max_grad_norm: float = 0.5
    # Commitment loss coefficient
    commitment_loss_coef: float = 0.25 # Beta value to weight the commitment loss
    # Codebook usage loss coefficient to encourage diversity
    code_usage_loss_coef: float = 0.1
    # Momentum rate for updating the target network in JEPA
    target_update_rate: float = 0.995

@dataclass
class ILConfig:
    """Configuration for the Iterated Learning process."""
    num_generations: int = 10
    # Steps for the initial warmup (Generation 0)
    warmup_steps: int = 50_000
    # Steps for the student to learn from a frozen teacher in each generation
    student_steps: int = 50_000
    # Steps of experience to collect with the new student policy for teacher refinement
    teacher_refinement_collect_steps: int = 10_000
    # Number of training updates to perform on the teacher model using the collected data
    teacher_refinement_updates: int = 10_000

@dataclass
class MainConfig:
    """Main configuration for the project."""
    project_name: str = "world_model_ratchet"
    run_name: Optional[str] = None
    experiment_dir: str = "experiments"
    run_timestamp: Optional[float] = None
    run_uuid: Optional[str] = None

    env: EnvConfig = field(default_factory=EnvConfig)

    # Perception 
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    # ActionAgent
    action: ActionConfig = field(default_factory=ActionConfig)
    # WorldModel
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    # ReplayBuffer
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    # TrainingConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # IteratedLearnerConfig
    il: ILConfig = field(default_factory=ILConfig)

def get_base_config():
    return MainConfig()