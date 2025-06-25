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
    def __init__(self, cfg: MainConfig):
        self.cfg = cfg
        self.device = self.cfg.training.device
        print(f"Using device: {self.device}")

        # --- Initialize Components ---
        self.env = CrafterEnvWrapper(cfg.env)
        
        state_dim = cfg.perception.code_dim
        num_actions = cfg.action.num_actions

        self.perception_agent = PerceptionAgent(cfg.perception).to(self.device)
        self.world_model = WorldModel(state_dim=state_dim, num_actions=num_actions, cfg=cfg.world_model).to(self.device)
        self.actor_critic = ActorCritic(state_dim=state_dim, cfg=cfg.action).to(self.device)

        # --- Initialize Optimizers ---
        world_model_params = list(self.perception_agent.parameters()) + list(self.world_model.parameters())
        self.world_optimizer = optim.Adam(world_model_params, lr=cfg.training.world_model_lr)
        self.action_optimizer = optim.Adam(self.actor_critic.parameters(), lr=cfg.training.action_model_lr)

        self.replay_buffer = ReplayBuffer(cfg.replay_buffer)
        
        # --- State Tracking ---
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_length = 0
        # We must initialize the first observation here
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
                state_representation = self.perception_agent(obs_batch)
                action, _ = self.actor_critic.get_action(state_representation)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.replay_buffer.add(self.obs, action, reward, next_obs, terminated, truncated)
            
            self.obs = next_obs
            self.episode_reward += reward
            self.episode_length += 1
            self.total_steps += 1

            if terminated or truncated:
                pbar.set_description(f"Ep Done | R: {self.episode_reward:.2f} | L: {self.episode_length}")
                self.obs, _ = self.env.reset()
                self.episode_reward = 0
                self.episode_length = 0

            # --- Learning ---
            if self.total_steps > self.cfg.training.learning_starts:
                batch = self.replay_buffer.sample(self.cfg.training.batch_size, self.device)
                if batch is not None:
                    # Pass the frozen flag to the update method
                    self.update_models(batch, teacher_is_frozen)

    def update_models(self, batch: dict, teacher_is_frozen: bool):
        """
        Performs a single update step on the models using a batch of data.
        """
        # --- World Model (JEPA) Update ---
        if not teacher_is_frozen:
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
            
        # --- 2. Actor-Critic (A2C) Update ---
        
        # Get necessary data from the batch
        obs_seq = batch['obs']
        actions_seq = batch['actions']
        rewards_seq = batch['rewards']
        dones_seq = batch['dones']
        next_obs_seq = batch['next_obs']
        batch_size, seq_len, _, _, _ = obs_seq.shape

        # We need state representations `z` for the A2C update.
        # We re-compute them here within a no_grad context because the A2C loss
        # should not flow back into the perception agent. The perception agent is
        # trained exclusively by the world model's JEPA loss.
        with torch.no_grad():
            obs_flat = obs_seq.reshape(batch_size * seq_len, *obs_seq.shape[2:])
            z_seq = self.perception_agent(obs_flat).reshape(batch_size, seq_len, -1)
        
        # Get action logits and state values from the Actor-Critic model
        action_logits_seq, state_values_seq = self.actor_critic(z_seq)
        state_values_seq = state_values_seq.squeeze(-1) # Shape: (batch_size, seq_len)

        # Create a distribution object to calculate log probabilities and entropy
        dist = torch.distributions.Categorical(logits=action_logits_seq)
        log_probs_seq = dist.log_prob(actions_seq) # Shape: (batch_size, seq_len)
        
        # --- Calculate Advantages and Returns (using N-step returns) ---
        with torch.no_grad():
            # Get the value of the state that comes *after* the final action in the sequence
            # This is used for bootstrapping the return calculation.
            last_next_obs = next_obs_seq[:, -1]
            last_z_next = self.perception_agent(last_next_obs)
            last_value = self.actor_critic.get_value(last_z_next).squeeze(-1)
            
            advantages = torch.zeros_like(rewards_seq)
            # A more advanced GAE would have a separate `last_gae_lam = 0` here.
            
            # Loop backwards through the sequence to calculate advantages
            for t in reversed(range(seq_len)):
                # mask = 1.0 if not done, 0.0 if done
                mask = 1.0 - dones_seq[:, t].float()
                
                # Determine the value of the next state (V(s_t+1))
                # If we are at the end of the sequence (t == seq_len - 1), the next value
                # is the `last_value` we bootstrapped. Otherwise, it's the next value
                # from our calculated sequence of values.
                next_value = state_values_seq[:, t + 1] if t < seq_len - 1 else last_value
                
                # Calculate the TD Error: R_t + gamma * V(s_t+1) - V(s_t)
                delta = rewards_seq[:, t] + self.cfg.training.gamma * next_value * mask - state_values_seq[:, t]
                
                # In this simple A2C implementation, the advantage is just the TD error.
                # A GAE implementation would add a discounted next advantage:
                # advantages[:, t] = delta + self.cfg.training.gamma * self.cfg.training.gae_lambda * mask * last_gae_lam
                advantages[:, t] = delta
                # last_gae_lam = advantages[:, t] # for GAE
        
        # The 'returns' are the target for the critic's value function.
        # It's the calculated advantage plus the original value estimate.
        # We detach state_values_seq because we don't want to backpropagate through it here.
        returns = advantages + state_values_seq.detach()
        
        # --- Calculate Final Losses ---
        # Actor Loss: a.k.a. Policy Gradient Loss. We want to increase the log_prob of
        # actions that had a high advantage. Detach advantages so we don't backprop through the critic.
        actor_loss = -(log_probs_seq * advantages.detach()).mean()

        # Critic Loss: a.k.a. Value Loss. This is a simple regression task to make the
        # critic's output (state_values_seq) match the calculated returns.
        critic_loss = F.mse_loss(state_values_seq, returns)
        
        # Entropy Loss: A bonus to encourage exploration by preventing the policy
        # from becoming too deterministic too quickly. We want to maximize entropy.
        entropy_loss = -dist.entropy().mean()

        # Combine the losses into a single objective for the action_optimizer
        total_action_loss = (actor_loss + 
                             self.cfg.training.critic_loss_coef * critic_loss + 
                             self.cfg.training.entropy_coef * entropy_loss)

        # --- Backpropagation for the Actor-Critic ---
        self.action_optimizer.zero_grad()
        total_action_loss.backward()
        # Gradient clipping can help stabilize training
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.training.max_grad_norm)
        self.action_optimizer.step()
