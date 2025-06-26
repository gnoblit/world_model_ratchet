import torch
import torch.nn as nn 
import torch.nn.functional as F
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

        # --- Create a separate, momentum-updated target perception agent for JEPA ---
        self.target_perception_agent = copy.deepcopy(self.perception_agent).to(self.device)
        # Freeze the target network; we will update it manually via Polyak averaging
        for param in self.target_perception_agent.parameters():
            param.requires_grad = False

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
        for _ in pbar:
            batch = self.replay_buffer.sample(self.cfg.training.batch_size, self.device)
            if batch is not None:
                # In this phase, we ONLY train the teacher.
                loss_dict = self.update_models(batch, teacher_is_frozen=False, student_is_frozen=True)
                self.total_grad_updates += 1 # Increment gradient update counter

                if self.logger:
                    # Log these refinement-specific losses against the *current* environment step count.
                    # This creates a "vertical" line of points on the TensorBoard graph, showing
                    # the effect of refinement at that point in experience.
                    # An alternative would be to log against total_grad_updates, but env_steps is more standard.
                    for key, value in loss_dict.items():
                        if key in ['world_model_loss', 'prediction_loss', 'codebook_loss', 'commitment_loss', 'code_entropy']:
                            self.logger.log_scalar(f'teacher_refinement/{key}', value, self.total_env_steps)

    @torch.no_grad()
    def _update_target_network(self):
        """Update the target network with a momentum-based (Polyak) average."""
        tau = self.cfg.training.target_update_rate
        for param, target_param in zip(self.perception_agent.parameters(), self.target_perception_agent.parameters()):
            target_param.data.copy_(tau * target_param.data + (1.0 - tau) * param.data)

    def update_models(self, batch: dict, teacher_is_frozen: bool, student_is_frozen: bool = False):
        """
        Performs a single update step on the models using a batch of data.
        """
        loss_dict = {}

        # Get data from the batch once at the top
        obs_seq = batch['obs']
        actions_seq = batch['actions']
        rewards_seq = batch['rewards']
        old_log_probs_seq = batch['log_probs']
        # Use 'terminateds' for value bootstrapping. 'truncateds' are handled by the mask.
        terminateds_seq = batch['terminateds']
        # --- FIX: Load the stored state representations from the buffer ---
        z_seq_stored = batch['state_reprs']
        next_obs_seq = batch['next_obs']
        batch_size, seq_len, C, H, W = obs_seq.shape

        # Reshape all sequences into flat batches for efficient processing
        obs_flat = obs_seq.reshape(batch_size * seq_len, C, H, W)
        next_obs_flat = next_obs_seq.reshape(batch_size * seq_len, C, H, W)
        actions_flat = actions_seq.reshape(batch_size * seq_len)
        
        # --- World Model / Perception Agent Path ---
        # We re-compute the representation z' from raw pixels using the *current* perception_agent.
        # This is necessary for the representation learning gradient.
        z_current_new, codebook_loss, commitment_loss, code_entropy = self.perception_agent(obs_flat)
        z_next_online, _, _, _ = self.perception_agent(next_obs_flat)

        # --- 1. World Model (JEPA) Update ---
        if not teacher_is_frozen:
            with torch.no_grad():
                # The prediction target comes from the frozen, momentum-updated target network.
                # This prevents the model from predicting its own immediate output (representational collapse).
                z_next_target, _, _, _ = self.target_perception_agent(next_obs_flat)
            # --------------------------------------------------------------------
            
            # Predict the next state using the *newly computed* representation z'
            z_next_predicted = self.world_model(z_current_new, actions_flat)
            
            prediction_loss = F.mse_loss(z_next_predicted, z_next_target)
            
            # The total world model loss is a combination of the prediction loss,
            # the VQ losses (codebook and commitment), and an entropy bonus.
            beta = self.cfg.training.commitment_loss_coef
            eta = self.cfg.training.code_usage_loss_coef
            # We subtract the entropy because we want to maximize it.
            world_model_loss = prediction_loss + codebook_loss + beta * commitment_loss - eta * code_entropy

            # Backpropagation for the world model
            self.world_optimizer.zero_grad()
            world_model_loss.backward()
            self.world_optimizer.step()

            # Log these specific losses
            loss_dict['world_model_loss'] = world_model_loss.item()
            loss_dict['prediction_loss'] = prediction_loss.item()
            loss_dict['codebook_loss'] = codebook_loss.item()
            loss_dict['commitment_loss'] = commitment_loss.item()
            loss_dict['code_entropy'] = code_entropy.item()

        if not student_is_frozen:
            # --- 2. Actor-Critic (A2C) Update ---
            # --- FIX: Use the STORED state representation `z_seq_stored` for the actor-critic update ---
            # This `z_seq_stored` is consistent with `old_log_probs_seq` from the buffer.
            # It is already detached from any graph as it comes from numpy.

            with torch.no_grad():
                # For the bootstrap value, use the *newly computed* representation of the next state.
                # This is a standard practice and is more accurate as it reflects the latest world understanding.
                # last_z_next is the representation of the last next_obs in the sequence,
                # derived from the current perception agent.
                last_z_next = z_next_online.reshape(batch_size, seq_len, -1)[:, -1].detach()
                last_value = self.actor_critic.get_value(last_z_next).squeeze(-1)

            # Get action logits and state values using the STORED state representation
            action_logits_seq, state_values_seq = self.actor_critic(z_seq_stored)
            state_values_seq = state_values_seq.squeeze(-1)

            dist = torch.distributions.Categorical(logits=action_logits_seq)
            new_log_probs_seq = dist.log_prob(actions_seq)
            
            # --- Calculate TD Targets and Advantages (GAE) ---
            # Both calculations are done within no_grad as they serve as fixed targets/baselines.
            with torch.no_grad():
                # Calculate next_values for the entire sequence, including the bootstrap value
                # next_values_for_gae will be (batch_size, seq_len)
                next_values_for_gae = torch.cat((state_values_seq[:, 1:], last_value.unsqueeze(-1)), dim=1)
                
                # The mask should only be applied for true terminal states, not truncated ones.
                # This is critical for correct value bootstrapping.
                masks = 1.0 - terminateds_seq.float()

                # Bug 4 Fix: Rename 'returns' to 'td_targets'
                # This is the 1-step TD target for the critic loss
                td_targets = rewards_seq + self.cfg.training.gamma * next_values_for_gae * masks
                
                # Bug 3 Fix: Implement GAE for advantages
                advantages = torch.zeros_like(rewards_seq) # (batch_size, seq_len)
                last_gae_lam = torch.zeros(batch_size, device=self.device) # (batch_size,)

                for t in reversed(range(seq_len)):
                    # Calculate 1-step TD error (delta)
                    delta = rewards_seq[:, t] + self.cfg.training.gamma * next_values_for_gae[:, t] * masks[:, t] - state_values_seq[:, t]
                    
                    # GAE formula: A_t = delta_t + gamma * lambda * A_{t+1} * mask_t
                    # The mask is crucial here to correctly handle episode boundaries
                    advantages[:, t] = delta + self.cfg.training.gamma * self.cfg.training.gae_lambda * last_gae_lam * masks[:, t]
                    last_gae_lam = advantages[:, t]
            
            # --- Advantage Normalization ---
            # A standard and critical step in PPO to stabilize training. We normalize
            # the advantages across the entire batch.
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # --- Calculate Final Losses ---
            # Importance sampling for off-policy correction.
            # We clip the ratio to stabilize training (PPO-style).
            eps = self.cfg.training.importance_clip_eps
            ratio = (new_log_probs_seq - old_log_probs_seq).exp()
            surr1 = ratio * advantages.detach()
            # The PPO ratio is now mathematically valid:
            # ratio = exp( log(π_new(a|z_stored)) - log(π_old(a|z_stored)) )

            # Note: old_log_probs_seq are from the replay buffer and were generated by a policy
            # acting on a state representation from an older version of the perception_agent.
            # This is a known trade-off in world-model-based RL where the state representation
            # itself evolves. The PPO clipping helps mitigate the variance from this mismatch.

            surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values_seq, td_targets) # Bug 4: Use td_targets
            entropy_loss = -dist.entropy().mean()
            total_action_loss = (actor_loss + 
                                self.cfg.training.critic_loss_coef * critic_loss + 
                                self.cfg.training.entropy_coef * entropy_loss)

            # --- Backpropagation for the Actor-Critic ---
            self.action_optimizer.zero_grad()
            # Crucially, because we used `z_seq_stored` (which has no grad_fn),
            # this loss will NOT backpropagate into the perception agent, which is correct
            # for the actor-critic update.
            total_action_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.training.max_grad_norm)
            self.action_optimizer.step()

            # Log A2C losses
            loss_dict['actor_loss'] = actor_loss.item()
            loss_dict['critic_loss'] = critic_loss.item()
            loss_dict['entropy_loss'] = entropy_loss.item()
            loss_dict['total_action_loss'] = total_action_loss.item()
        
        # After the optimizer steps, update the target network if the teacher is training
        if not teacher_is_frozen:
            self._update_target_network()

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
