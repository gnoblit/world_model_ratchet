import torch
import torch.nn as nn 
import torch.nn.functional as F
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
        self.actor_critic = ActorCritic(state_dim=state_dim, cfg=cfg.action).to(self.device)

        # --- Init Optimizers ---
        world_model_params = list(self.perception_agent.parameters()) + list(self.world_model.parameters())
        self.world_optimizer = optim.Adam(world_model_params, lr=cfg.training.world_model_lr)
        self.action_optimizer = optim.Adam(self.actor_critic.parameters(), lr=cfg.training.action_model_lr)

        self.replay_buffer = ReplayBuffer(cfg.replay_buffer)
        
        # Init Logger
        self.logger = None
        self.log_dir = None

        # Only initialize the logger if a run_name is provided
        if cfg.run_name:
            self.log_dir = os.path.join(cfg.experiment_dir, cfg.run_name)
            self.logger = Logger(self.log_dir)

        # --- State Tracking ---
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_length = 0
        # We must init the first observation here
        self.obs, _ = self.env.reset()

        print("Trainer initialized successfully.")

    def train_for_steps(self, num_steps: int, teacher_is_frozen: bool = False):
        """
        Runs the training loop for a specific number of steps.

        Args:
            num_steps (int): The number of environment steps to run for.
            teacher_is_frozen (bool): If True, freezes the world model (teacher) updates.
        """
        pbar = tqdm(range(num_steps), desc=f"Training (Teacher Frozen: {teacher_is_frozen})")
        for _ in pbar:
            # --- Interaction ---
            obs_batch = self.obs.unsqueeze(0).to(self.device)
            with torch.no_grad():
                state_representation, _, _ = self.perception_agent(obs_batch)
                action, _ = self.actor_critic.get_action(state_representation)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.replay_buffer.add(self.obs, action, reward, next_obs, terminated, truncated)
            
            self.obs = next_obs
            self.episode_reward += reward
            self.episode_length += 1
            self.total_steps += 1

            pbar.set_postfix({
                'Reward': f"{self.episode_reward:.2f}", 
                'Length': self.episode_length,
                'Total Steps': self.total_steps
            })
            
            if terminated or truncated:
                # --- LOGGING EPISODE STATS ---
                if self.logger:
                    self.logger.log_scalar('rollout/episode_reward', self.episode_reward, self.total_steps)
                    self.logger.log_scalar('rollout/episode_length', self.episode_length, self.total_steps)


                self.obs, _ = self.env.reset()
                self.episode_reward = 0
                self.episode_length = 0

            if (self.total_steps > self.cfg.training.learning_starts and 
                self.total_steps % self.cfg.training.update_every_steps == 0):
                
                batch = self.replay_buffer.sample(self.cfg.training.batch_size, self.device)
                if batch is not None:
                    loss_dict = self.update_models(batch, teacher_is_frozen)
                    
                    if self.logger: # No need to check step count here, it's already periodic
                        for key, value in loss_dict.items():
                            self.logger.log_scalar(f'train/{key}', value, self.total_steps)
    
    def update_models(self, batch: dict, teacher_is_frozen: bool):
        """
        Performs a single update step on the models using a batch of data.
        """
        loss_dict = {}

        # Get data from the batch once at the top
        obs_seq = batch['obs']
        actions_seq = batch['actions']
        rewards_seq = batch['rewards']
        dones_seq = batch['dones']
        next_obs_seq = batch['next_obs']
        batch_size, seq_len, C, H, W = obs_seq.shape

        # Reshape all sequences into flat batches for efficient processing
        obs_flat = obs_seq.reshape(batch_size * seq_len, C, H, W)
        next_obs_flat = next_obs_seq.reshape(batch_size * seq_len, C, H, W)
        actions_flat = actions_seq.reshape(batch_size * seq_len)

        # --- 1. World Model (JEPA) Update ---
        if not teacher_is_frozen:
            # --- FIX: The gradients should only be disabled for the target network. ---
            # The main perception agent call needs to compute gradients.
            z_current, commitment_loss, code_entropy = self.perception_agent(obs_flat)

            with torch.no_grad():
                # We only need the representation for the target, not the losses
                z_next_target, _, _ = self.perception_agent(next_obs_flat)
            # --------------------------------------------------------------------

            z_next_predicted = self.world_model(z_current, actions_flat)
            
            prediction_loss = F.mse_loss(z_next_predicted, z_next_target) # .detach() is redundant as it's in no_grad
            
            # The total world model loss is a combination of the prediction loss,
            # the commitment loss (to keep the encoder honest), and an entropy
            # bonus (to encourage diverse codebook usage).
            beta = self.cfg.training.commitment_loss_coef
            eta = self.cfg.training.code_usage_loss_coef
            # We subtract the entropy because we want to maximize it.
            world_model_loss = prediction_loss + beta * commitment_loss - eta * code_entropy

            # Backpropagation for the world model
            self.world_optimizer.zero_grad()
            world_model_loss.backward()
            self.world_optimizer.step()

            # Log these specific losses
            loss_dict['world_model_loss'] = world_model_loss.item()
            loss_dict['prediction_loss'] = prediction_loss.item()
            loss_dict['commitment_loss'] = commitment_loss.item()
            loss_dict['code_entropy'] = code_entropy.item()

        # --- 2. Actor-Critic (A2C) Update ---
        with torch.no_grad():
            # --- FIX: We only need to compute z_seq here for the A2C loss. ---
            # If the teacher was frozen, z_current wasn't computed yet in the block above.
            # So we must compute it now.
            if teacher_is_frozen: 
                z_current, _, _ = self.perception_agent(obs_flat)
            
            # We can reuse z_current if it was already computed, but it will still have gradients.
            # We must detach it before reshaping for the A2C update.
            z_seq = z_current.detach().reshape(batch_size, seq_len, -1)
            
            # --- OPTIMIZATION: Avoid recomputing last_z_next if possible ---
            if not teacher_is_frozen:
                # z_next_target was computed for the world model loss.
                # We can slice the last state representation from it.
                indices = torch.arange(batch_size, device=self.device) * seq_len + (seq_len - 1)
                last_z_next = z_next_target[indices]
            else:
                # Otherwise, compute it from scratch
                last_next_obs = next_obs_seq[:, -1]
                last_z_next, _, _ = self.perception_agent(last_next_obs)

            last_value = self.actor_critic.get_value(last_z_next).squeeze(-1)

        # Get action logits and state values from the Actor-Critic model
        # We use z_seq (which is detached) to ensure no gradients flow from A2C back to the perception agent
        action_logits_seq, state_values_seq = self.actor_critic(z_seq)
        state_values_seq = state_values_seq.squeeze(-1)

        dist = torch.distributions.Categorical(logits=action_logits_seq)
        log_probs_seq = dist.log_prob(actions_seq)
        
        # --- Calculate Advantages and Returns ---
        # OPTIMIZATION: Vectorized advantage and return calculation
        with torch.no_grad():
            next_values = torch.cat((state_values_seq[:, 1:], last_value.unsqueeze(-1)), dim=1)
            masks = 1.0 - dones_seq.float()
            returns = rewards_seq + self.cfg.training.gamma * next_values * masks
            # Advantages are TD-errors in this case (r_t + gamma*V(s_{t+1}) - V(s_t))
            advantages = returns - state_values_seq
        
        # --- Calculate Final Losses ---
        actor_loss = -(log_probs_seq * advantages.detach()).mean()
        critic_loss = F.mse_loss(state_values_seq, returns)
        entropy_loss = -dist.entropy().mean()
        total_action_loss = (actor_loss + 
                            self.cfg.training.critic_loss_coef * critic_loss + 
                            self.cfg.training.entropy_coef * entropy_loss)

        # --- Backpropagation for the Actor-Critic ---
        self.action_optimizer.zero_grad()
        total_action_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.training.max_grad_norm)
        self.action_optimizer.step()

        # Log A2C losses
        loss_dict['actor_loss'] = actor_loss.item()
        loss_dict['critic_loss'] = critic_loss.item()
        loss_dict['entropy_loss'] = entropy_loss.item()
        loss_dict['total_action_loss'] = total_action_loss.item()
        
        return loss_dict
  
    def save_models(self):
        """Saves the current state of the models to the experiment directory."""
        if not self.log_dir:
            print("No log directory specified. Skipping model save.")
            return

        # Define paths for each model component
        perception_path = os.path.join(self.log_dir, "perception_agent.pth")
        actor_critic_path = os.path.join(self.log_dir, "actor_critic.pth")
        world_model_path = os.path.join(self.log_dir, "world_model.pth")
        
        # Save the state dict for each model
        torch.save(self.perception_agent.state_dict(), perception_path)
        torch.save(self.actor_critic.state_dict(), actor_critic_path)
        torch.save(self.world_model.state_dict(), world_model_path)
        
        print(f"Models saved to {self.log_dir}")
    
    def close(self):
        """A helper method to clean up resources and save models."""
        print("Closing trainer and saving models...")
        self.save_models() # Call save before closing
        self.env.close()
        if self.logger:
            self.logger.close()
