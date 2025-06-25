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

class Trainer:
    """The main trainer class to orchestrate training process."""
    def __init__(self, cfg):
        """
        Initializes the environment, models, optimizers, and replay buffer.
        """
        self.cfg = cfg
        self.device = self.cfg.training.device
        print(f"Using device: {self.device}")

        # 1. Init Environment
        self.env = CrafterEnvWrapper(cfg.env)

        # 2. Init models
        state_dim = cfg.perception.code_dim

        self.perception_agent = PerceptionAgent(cfg.perception).to(self.device)
        self.actor_critic  = ActorCritic(state_dim=state_dim, cfg=cfg.action).to(self.device)
        self.world_model = WorldModel(state_dim=state_dim, num_actions=cfg.action.num_actions, cfg=cfg.world_model).to(self.device)

        # 3. Init Optimizers
        world_model_params = list(self.perception_agent.parameters()) + list(self.world_model.parameters())
        
        self.world_optimizer = optim.Adam(world_model_params, lr=cfg.training.world_model_lr)
        self.action_optimizer = optim.Adam(self.actor_critic .parameters(), lr=cfg.training.action_model_lr)

        # 4. Init Replay Buffer
        self.replay_buffer = ReplayBuffer(cfg.replay_buffer)
        
        print("Trainer initialized successfully.")

    def train(self):
        """
        The main training loop.
        """
        print("Starting training...")
        
        # Reset the environment to get the first observation
        obs, _ = self.env.reset()
        
        # Use tqdm for a progress bar over the total training steps
        for step in tqdm(range(self.cfg.training.total_train_steps), desc="Training Progress"):
            
            # --- Phase 1: Interaction ---
            # Observation from env is already a tensor, so just add a batch_dim
            obs_batch = obs.unsqueeze(0).to(self.device)

            # Detach representation from computation graph
            # when using for action selection. This will prevent
            # gradients from flowing.
            with torch.no_grad():
                # 1. Get state representation, `z_t`, from perception_agent
                state_representation = self.perception_agent(obs_batch)

                # 2. Get action, `a_t` from actor_critic 
                action, _ = self.actor_critic .get_action(state_representation)
            
            # 3. Step environment with action
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # Add the transition to replay buffer (obs not batched)
            self.replay_buffer.add(obs, action, reward, next_obs, terminated, truncated)

            # Update state
            obs = next_obs
            episode_reward += reward
            episode_length += 1

            # If terminated/truncated, reset
            if terminated or truncated:
                print(f"\nEpisode finished. Steps: {episode_length}, Reward: {episode_reward:.2f}")
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0

            # --- Phase 2: Learning ---
            # Check if enough experience to learn
            if step > self.cfg.training.learning_starts:
                # Sample a batch of trajectories from the replay buffer
                batch = self.replay_buffer.sample(self.cfg.training.batch_size, self.device)

                if batch is not None:
                    # Update
                    self.update_models(batch)

            # TODO: Logging
            # Log metrics periodically (e.g., every 1000 steps)
            pass

        self.env.close()
        print("Training finished.")

    def update_models(self, batch: dict):
        """
        Performs a single update step on the models using a batch of data.
        """
        # --- 1. Calculate JEPA World Model Loss ---
        
        # Get the data from the batch
        # obs and next_obs have shape (batch_size, seq_len, C, H, W)
        # actions has shape (batch_size, seq_len)
        obs_seq = batch['obs']
        actions_seq = batch['actions']
        next_obs_seq = batch['next_obs']

        # We need to process the whole sequence. Reshape to process all steps at once.
        # This is more efficient than looping through the sequence.
        batch_size, seq_len, C, H, W = obs_seq.shape
        obs_flat = obs_seq.reshape(batch_size * seq_len, C, H, W)
        next_obs_flat = next_obs_seq.reshape(batch_size * seq_len, C, H, W)
        actions_flat = actions_seq.reshape(batch_size * seq_len)

        # Get the representations z_t and z_t+1
        # We use no_grad for the target representation z_t+1 to stabilize training.
        # This is a common practice in self-supervised learning (like BYOL).
        # It prevents the model from collapsing by chasing a moving target.
        with torch.no_grad():
            z_next_target = self.perception_agent(next_obs_flat)

        z_current = self.perception_agent(obs_flat)

        # Predict the next state's representation: z_hat_t+1
        z_next_predicted = self.world_model(z_current, actions_flat)
        
        # Calculate the loss: Mean Squared Error between predicted and target representations
        # We want to make the predicted vector as close as possible to the target vector.
        world_model_loss = F.mse_loss(z_next_predicted, z_next_target.detach())

        # Backpropagation for the world model
        self.world_optimizer.zero_grad()
        world_model_loss.backward()
        self.world_optimizer.step()
        
        # 2. --- 2. Actor-Critic (A2C) Loss ---

        # State representations for `z_t`, detached
        z_seq = z_current.detach().reshape(batch_size, seq_len, -1)

                # Get action logits and state values for the entire sequence
        # The .squeeze(-1) removes the trailing dimension from the critic's output
        action_logits_seq, state_values_seq = self.actor_critic(z_seq)
        state_values_seq = state_values_seq.squeeze(-1) # Shape: (batch_size, seq_len)

        # Create a categorical distribution to get log probabilities
        dist = Categorical(logits=action_logits_seq)
        log_probs_seq = dist.log_prob(actions_seq) # Shape: (batch_size, seq_len)
        
        # Calculate advantages and returns (we'll use simple 1-step returns for now)
        # A more advanced implementation would use Generalized Advantage Estimation (GAE)
        rewards_seq = batch['rewards']
        dones_seq = batch['dones']

        # The target for the value function is the immediate reward + discounted value of the next state
        with torch.no_grad():
            # Get the value of the final state in each sequence to bootstrap
            # We need the representation of the state AFTER the last action
            last_next_obs = next_obs_seq[:, -1] # Get the very last observation in each sequence
            last_z_next = self.perception_agent(last_next_obs)
            last_value = self.actor_critic.get_value(last_z_next).squeeze(-1)
            
            # Simplified N-step returns calculation
            advantages = torch.zeros_like(rewards_seq)
            last_advantage = 0
            for t in reversed(range(seq_len)):
                mask = 1.0 - dones_seq[:, t].float() # If done, the future value is 0
                
                # The "TD Error" or 1-step advantage
                delta = rewards_seq[:, t] + self.cfg.training.gamma * (state_values_seq[:, t + 1] if t < seq_len - 1 else last_value) * mask - state_values_seq[:, t]
                
                # For now, let's use the simple advantage (a proper GAE would be better)
                advantages[:, t] = delta # In this simple case, Advantage is the TD error
        
        # --- Calculate losses ---
        # Actor loss (Policy Gradient)
        actor_loss = -(log_probs_seq * advantages.detach()).mean()

        # Critic loss (MSE between predicted values and the calculated returns)
        # The target for the critic is the advantage + the value prediction
        returns = advantages + state_values_seq.detach()
        critic_loss = F.mse_loss(state_values_seq, returns)
        
        # Entropy loss for encouraging exploration
        entropy_loss = -dist.entropy().mean()

        # Total loss for the Actor-Critic
        total_action_loss = (actor_loss + 
                             self.cfg.training.critic_loss_coef * critic_loss + 
                             self.cfg.training.entropy_coef * entropy_loss)

        # Backpropagation for the Actor-Critic
        self.action_optimizer.zero_grad()
        total_action_loss.backward()
        self.action_optimizer.step()
