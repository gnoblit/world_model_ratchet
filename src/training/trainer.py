import torch
import torch.nn as nn 
import torch.nn.functional as F
import time # Added: Missing import for time.time()
import copy
import torch.optim as optim
from torch.distributions import Categorical

from tqdm import tqdm
    
from configs.base_config import MainConfig
from environment.crafter_env import CrafterEnvWrapper
from models.perception import PerceptionAgent
from models.actor_critic import ActorCritic
from models.world_model import WorldModel
from utils.replay_buffer import ReplayBuffer
from utils.logger import Logger

import os

class Trainer:
    def __init__(self, cfg: MainConfig):
        self.cfg = cfg
        self.device = self.cfg.training.device
        print(f"Using device: {self.device}")

        # --- Init Components ---
        self.env = CrafterEnvWrapper(cfg.env)
        
        state_dim = cfg.perception.code_dim
        num_actions = cfg.action.num_actions

        self.perception_agent = PerceptionAgent(cfg.perception).to(self.device)
        self.world_model = WorldModel(state_dim=state_dim, num_actions=num_actions, cfg=cfg.world_model).to(self.device)
        self.actor_critic = ActorCritic(state_dim=state_dim, num_actions=num_actions, cfg=cfg.action).to(self.device)

        # --- Compile models with torch.compile for a significant speedup ---
        # This is a major performance boost in PyTorch 2.0+. It JIT-compiles the model
        # graph into optimized kernels.
        if cfg.training.use_torch_compile and self.device == 'cuda':
            print("Compiling models with torch.compile...")
            # Using 'reduce-overhead' mode is a good balance for dynamic shapes.
            # 'max-autotune' is more aggressive but can have a long warmup.
            self.perception_agent = torch.compile(self.perception_agent, mode="reduce-overhead")
            self.world_model = torch.compile(self.world_model, mode="reduce-overhead")
            self.actor_critic = torch.compile(self.actor_critic, mode="reduce-overhead")
            print("Models compiled successfully.")

        # --- Create a separate, momentum-updated target perception agent for JEPA ---
        # CRITICAL FIX: Deepcopy *after* potential torch.compile.
        # This ensures the target network is also a compiled module if compilation is enabled.
        self.target_perception_agent = copy.deepcopy(self.perception_agent).to(self.device)
        # Freeze the target network; we will update it manually via Polyak averaging
        for param in self.target_perception_agent.parameters():
            param.requires_grad = False

        # --- Init Optimizers (FIX: Separate optimizers for teacher components) ---
        # Using separate optimizers allows for different learning rates for the large
        # perception backbone and the smaller world model head, which is a common practice.
        self.perception_optimizer = optim.Adam(self.perception_agent.parameters(), lr=cfg.training.perception_model_lr)
        self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=cfg.training.world_model_lr)
        self.action_optimizer = optim.Adam(self.actor_critic.parameters(), lr=cfg.training.action_model_lr)

        # --- Mixed Precision Scaler ---
        # If using cuda, GradScaler helps prevent underflow with fp16 gradients.
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

        self.replay_buffer = ReplayBuffer(cfg.replay_buffer)
        
        # Init Logger
        self.logger = None
        self.log_dir = None

        # Only initialize the logger if a run_name is provided
        if cfg.run_name:
            self.log_dir = os.path.join(cfg.experiment_dir, cfg.run_name)
            self.logger = Logger(self.log_dir)

        # --- State Tracking ---
        self.total_env_steps = 0 # Total environment steps taken
        self.total_grad_updates = 0 # Total gradient updates performed across all models
        self.episode_reward = 0
        self.episode_length = 0
        # We must init the first observation here
        self.obs, _ = self.env.reset()

        print("Trainer initialized successfully.")

    def _step_env(self):
        """Takes a single step in the environment, adds to buffer, and handles episode termination."""
        # --- Interaction ---
        obs_batch = self.obs.unsqueeze(0).to(self.device)
        with torch.no_grad():
            # state_representation is a (1, state_dim) tensor here. Squeeze it for buffer storage.
            state_representation, _, _, _ = self.perception_agent(obs_batch)
            state_representation_for_buffer = state_representation.squeeze(0)
            # Get action and its log probability from the current policy
            action_tensor, log_prob_tensor = self.actor_critic.get_action(state_representation)

        action = action_tensor.item()
        log_prob = log_prob_tensor.item()
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.replay_buffer.add(self.obs, action, log_prob, reward, next_obs, terminated, truncated, state_representation_for_buffer)
        
        self.obs = next_obs
        self.episode_reward += reward
        self.episode_length += 1
        self.total_env_steps += 1
        
        if terminated or truncated:
            if self.logger:
                self.logger.log_scalar('rollout/episode_reward', self.episode_reward, self.total_env_steps)
                self.logger.log_scalar('rollout/episode_length', self.episode_length, self.total_env_steps)
            self.obs, _ = self.env.reset()
            self.episode_reward = 0
            self.episode_length = 0

    def collect_experience(self, num_env_steps: int):
        """Collects experience by interacting with the environment without training."""
        pbar = tqdm(range(num_env_steps), desc="Collecting Experience")
        for _ in pbar:
            self._step_env()
            pbar.set_postfix({
                'Reward': f"{self.episode_reward:.2f}", 
                'Length': self.episode_length,
                'Env Steps': self.total_env_steps
            })

    def train_for_steps(self, num_env_steps: int, teacher_is_frozen: bool = False):
        """
        Runs the training loop for a specific number of environment steps.

        Args:
            num_env_steps (int): The number of environment steps to run for.
            teacher_is_frozen (bool): If True, freezes the world model (teacher) updates.
        """
        pbar = tqdm(range(num_env_steps), desc=f"Training (Teacher Frozen: {teacher_is_frozen})")
        for _ in pbar:
            self._step_env()
            pbar.set_postfix({
                'Reward': f"{self.episode_reward:.2f}", 
                'Length': self.episode_length,
                'Env Steps': self.total_env_steps
            })
            
            if (self.total_env_steps > self.cfg.training.learning_starts and 
                self.total_env_steps % self.cfg.training.update_every_steps == 0):
                
                batch = self.replay_buffer.sample(self.cfg.training.batch_size, self.device)
                if batch is not None:
                    # In this interactive loop, the student is always training.
                    loss_dict = self.update_models(batch, teacher_is_frozen, student_is_frozen=False)
                    self.total_grad_updates += 1 # Increment gradient update counter
                    
                    if self.logger: # No need to check step count here, it's already periodic
                        for key, value in loss_dict.items():
                            self.logger.log_scalar(f'train/{key}', value, self.total_env_steps)
    
    def train_from_buffer(self, num_updates: int):
        """Trains models from the replay buffer without environment interaction."""
        pbar = tqdm(range(num_updates), desc="Refining Teacher from Buffer")
        for i in pbar:
            batch = self.replay_buffer.sample(self.cfg.training.batch_size, self.device)
            if batch is not None:
                # In this phase, we ONLY train the teacher.
                loss_dict = self.update_models(batch, teacher_is_frozen=False, student_is_frozen=True)
                self.total_grad_updates += 1 # Increment gradient update counter

                if self.logger:
                    # --- FIX: Log against fractional env steps for better visualization ---
                    # To visualize the trend during refinement without creating a separate x-axis,
                    # we log against a fractional step count. This plots the refinement progress
                    # within a single step on the main timeline, e.g., between step 100000 and 100001.
                    log_step = self.total_env_steps + (i / num_updates)
                    for key, value in loss_dict.items():
                        if key in ['world_model_loss', 'prediction_loss', 'codebook_loss', 'commitment_loss', 'code_entropy']:
                            self.logger.log_scalar(f'teacher_refinement/{key}', value, log_step)

    @torch.no_grad()
    def _update_target_network(self):
        """Update the target network with a momentum-based (Polyak) average."""
        tau = self.cfg.training.target_update_rate
        for param, target_param in zip(self.perception_agent.parameters(), self.target_perception_agent.parameters()):
            target_param.data.copy_(tau * target_param.data + (1.0 - tau) * param.data)
    
    def update_models(self, batch: dict, teacher_is_frozen: bool, student_is_frozen: bool = False):
        """
        Performs a single update step on the models using a batch of data.
        This method now uses `torch.cuda.amp.autocast` for mixed-precision training
        and a `GradScaler` for safe backpropagation. The forward passes are performed
        in the `autocast` context, and the backward passes and optimizer steps are
        handled outside with the scaler.
        """
        loss_dict = {}
        update_start_time = time.time() # Add this line for timing
        world_model_loss, total_action_loss = None, None

        # Use autocast for the forward passes to enable mixed precision.
        # All tensor operations within this block are automatically cast to fp16 on CUDA devices.
        with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
            # Get data from the batch once at the top
            obs_seq = batch['obs']
            actions_seq = batch['actions']
            rewards_seq = batch['rewards']
            old_log_probs_seq = batch['log_probs']
            terminateds_seq = batch['terminateds'] 
            z_seq_stored = batch['state_reprs']
            next_obs_seq = batch['next_obs']
            batch_size, seq_len, C, H, W = obs_seq.shape

            # --- VRAM FIX: Process sequences step-by-step instead of flattening ---
            z_next_target_list, z_next_predicted_list = [], []
            codebook_loss_list, commitment_loss_list, code_entropy_list = [], [], []

            # --- World Model and VQ Loss Calculation (only if teacher is training) ---
            if not teacher_is_frozen:
                for t in range(seq_len):
                    # --- CRITICAL FIX: Use stored state representations for world model ---
                    # The world model must learn the dynamics of the same representation space
                    # that the actor-critic uses (z_seq_stored).
                    z_t_stored = z_seq_stored[:, t]
                    actions_t = actions_seq[:, t]
                    z_next_predicted_t = self.world_model(z_t_stored, actions_t)
                    z_next_predicted_list.append(z_next_predicted_t)

                    # The VQ losses are still needed to train the perception agent itself.
                    # We compute them here but the representation output is not used for the WM/AC.
                    _, cb_loss_t, cmt_loss_t, entr_t = self.perception_agent(obs_seq[:, t], compute_losses=True)
                    codebook_loss_list.append(cb_loss_t)
                    commitment_loss_list.append(cmt_loss_t)
                    code_entropy_list.append(entr_t)

                    # The target for the world model is the representation of the next observation.
                    with torch.no_grad():
                        z_next_target_t, _, _, _ = self.target_perception_agent(next_obs_seq[:, t], compute_losses=False)
                    z_next_target_list.append(z_next_target_t)
                            
            # --- 1. World Model (JEPA) Loss Calculation ---
            if not teacher_is_frozen:
                codebook_loss = torch.stack(codebook_loss_list).mean()
                commitment_loss = torch.stack(commitment_loss_list).mean()
                code_entropy = torch.stack(code_entropy_list).mean()
                z_next_predicted = torch.stack(z_next_predicted_list, dim=1)
                z_next_target = torch.stack(z_next_target_list, dim=1)
                prediction_loss = F.mse_loss(z_next_predicted, z_next_target)
                
                beta = self.cfg.training.commitment_loss_coef
                eta = self.cfg.training.code_usage_loss_coef
                world_model_loss = prediction_loss + codebook_loss + beta * commitment_loss - eta * code_entropy

                loss_dict['world_model_loss'] = world_model_loss.item()
                loss_dict['prediction_loss'] = prediction_loss.item()
                loss_dict['codebook_loss'] = codebook_loss.item()
                loss_dict['commitment_loss'] = commitment_loss.item()
                loss_dict['code_entropy'] = code_entropy.item()

            # --- 2. Actor-Critic (A2C) Loss Calculation ---
            if not student_is_frozen:
                with torch.no_grad():
                    last_next_obs = next_obs_seq[:, -1]
                    last_z_next, _, _, _ = self.perception_agent(last_next_obs)
                    last_value = self.actor_critic.get_value(last_z_next).squeeze(-1)
                
                # --- CRITICAL BUG FIX: Use the stored state representations from the buffer ---
                # This ensures the PPO importance sampling ratio is mathematically valid, as
                # both old and new log probabilities are conditioned on the same state input.
                action_logits_seq, state_values_seq = self.actor_critic(z_seq_stored)
                state_values_seq = state_values_seq.squeeze(-1)
                dist = torch.distributions.Categorical(logits=action_logits_seq)
                new_log_probs_seq = dist.log_prob(actions_seq)
                
                with torch.no_grad():
                    next_values_for_gae = torch.cat((state_values_seq[:, 1:], last_value.unsqueeze(-1)), dim=1)
                    masks = 1.0 - terminateds_seq.float()
                    advantages = torch.zeros_like(rewards_seq)
                    last_gae_lam = torch.zeros(batch_size, device=self.device)
                    for t in reversed(range(seq_len)):
                        delta = rewards_seq[:, t] + self.cfg.training.gamma * next_values_for_gae[:, t] * masks[:, t] - state_values_seq[:, t]
                        advantages[:, t] = delta + self.cfg.training.gamma * self.cfg.training.gae_lambda * last_gae_lam * masks[:, t]
                        last_gae_lam = advantages[:, t]
                    returns = advantages + state_values_seq

                advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                eps = self.cfg.training.importance_clip_eps
                ratio = (new_log_probs_seq - old_log_probs_seq).exp()
                surr1 = ratio * advantages_normalized.detach()
                surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages_normalized.detach()
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(state_values_seq, returns.detach())
                entropy_loss = -dist.entropy().mean()
                total_action_loss = (actor_loss + 
                                    self.cfg.training.critic_loss_coef * critic_loss + 
                                    self.cfg.training.entropy_coef * entropy_loss)

                loss_dict['actor_loss'] = actor_loss.item()
                loss_dict['critic_loss'] = critic_loss.item()
                loss_dict['entropy_loss'] = entropy_loss.item()
                loss_dict['total_action_loss'] = total_action_loss.item()
        
        # --- Backpropagation and Optimizer Steps (outside autocast) ---
        # We use the GradScaler to prevent underflow of fp16 gradients.
        # We combine losses before the backward pass to handle shared graph components correctly.
        combined_loss = 0
        if not teacher_is_frozen and world_model_loss is not None:
            combined_loss += world_model_loss

        if not student_is_frozen and total_action_loss is not None:
            combined_loss += total_action_loss

        # Only perform backprop and update if there's a loss to optimize
        if isinstance(combined_loss, torch.Tensor):
            # Zero-out all gradients first
            self.perception_optimizer.zero_grad(set_to_none=True)
            self.world_optimizer.zero_grad(set_to_none=True)
            self.action_optimizer.zero_grad(set_to_none=True)

            # Scale the combined loss and perform a single backward pass
            self.scaler.scale(combined_loss).backward()

            # Unscale and clip gradients only for optimizers that will be stepped
            if not teacher_is_frozen:
                self.scaler.unscale_(self.perception_optimizer)
                self.scaler.unscale_(self.world_optimizer)
                torch.nn.utils.clip_grad_norm_(self.perception_agent.parameters(), self.cfg.training.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.cfg.training.max_grad_norm)
            if not student_is_frozen:
                self.scaler.unscale_(self.action_optimizer)
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.training.max_grad_norm)

            # Step the optimizers for the models that are not frozen
            if not teacher_is_frozen:
                self.scaler.step(self.perception_optimizer)
                self.scaler.step(self.world_optimizer)
            if not student_is_frozen:
                self.scaler.step(self.action_optimizer)
        
        # Update the scaler once after all optimizer steps for this iteration
        self.scaler.update()

        # After the optimizer steps, update the target network if the teacher is training
        if not teacher_is_frozen:
            self._update_target_network()

        # Log the time taken for the update
        update_duration = time.time() - update_start_time
        loss_dict['update_duration_ms'] = update_duration * 1000 # Milliseconds

        return loss_dict

    def save_models(self, suffix: str = ""):
        """
        Saves the current state of the models to the experiment directory.
        Args:
            suffix (str): A suffix to append to the model filenames (e.g., "_gen_1").
        """
        if not self.log_dir:
            print("No log directory specified. Skipping model save.")
            return

        # Add suffix to filenames if provided, before the extension
        perception_filename = f"perception_agent{suffix}.pth"
        actor_critic_filename = f"actor_critic{suffix}.pth"
        world_model_filename = f"world_model{suffix}.pth"

        # Define paths for each model component
        perception_path = os.path.join(self.log_dir, perception_filename)
        actor_critic_path = os.path.join(self.log_dir, actor_critic_filename)
        world_model_path = os.path.join(self.log_dir, world_model_filename)
        
        # Save the state dict for each model
        torch.save(self.perception_agent.state_dict(), perception_path)
        torch.save(self.actor_critic.state_dict(), actor_critic_path)
        torch.save(self.world_model.state_dict(), world_model_path)
        
        print(f"Models saved to {self.log_dir} (suffix: '{suffix}')")
    
    def close(self):
        """A helper method to clean up resources and save models."""
        print("Closing trainer and saving models...")
        self.save_models(suffix="_final") # Call save before closing
        self.env.close()
        if self.logger:
            self.logger.close()
