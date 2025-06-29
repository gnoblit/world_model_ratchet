import torch
import numpy as np
import random
from collections import deque

class ReplayBuffer:
  """
  A replay buffer that stores entire trajectories as dictionaries of NumPy arrays
  and samples sequences of transitions. This approach is more memory-efficient 
  compared to storing individual transitions, especially for longer episodes.
  """
  def __init__(self, cfg):
      self.capacity = cfg.capacity # Max number of *episodes* to store
      self.sequence_length = cfg.sequence_length
      self.buffer = deque(maxlen=self.capacity)
      # --- OPTIMIZATION: Add a separate deque for episodes valid for sampling ---
      self.valid_buffer = deque(maxlen=self.capacity)
      self.reset_current_episode()

  def reset_current_episode(self):
      """Resets the temporary buffers for the current episode."""
      self.current_episode = {
          'obs': [], 'actions': [], 'log_probs': [], 'rewards': [], 
          'next_obs': [], 'terminateds': [], 'truncateds': [], 'state_reprs': []
      }

  def add(self, obs, action, log_prob, reward, next_obs, terminated, truncated, state_repr):
      """Adds a single transition to the current episode's temporary lists."""
      # Convert float32 CHW tensors back to uint8 HWC numpy arrays for memory efficiency.
      # The environment wrapper's ToTensor() transform handles the reverse process during sampling.
      # This saves significant RAM by storing pixels as uint8 instead of float32.
      obs_uint8 = (obs.cpu().permute(1, 2, 0) * 255).to(torch.uint8).numpy()
      next_obs_uint8 = (next_obs.cpu().permute(1, 2, 0) * 255).to(torch.uint8).numpy()

      self.current_episode['obs'].append(obs_uint8)
      self.current_episode['actions'].append(action)
      self.current_episode['log_probs'].append(log_prob)
      self.current_episode['rewards'].append(reward)
      self.current_episode['next_obs'].append(next_obs_uint8)
      # Store terminated and truncated flags separately to correctly handle
      # value bootstrapping for truncated episodes.
      self.current_episode['state_reprs'].append(state_repr.cpu().numpy())
      self.current_episode['terminateds'].append(terminated)
      self.current_episode['truncateds'].append(truncated)
      if terminated or truncated:
          self.commit_episode()

  def commit_episode(self):
      """Converts the current episode lists to NumPy arrays and stores them."""
      if not self.current_episode['actions']:
          return # Don't commit empty episodes

      # --- CRITICAL BUG FIX: Robust eviction synchronization ---
      # If the main buffer is full, appending a new episode will cause the oldest
      # one to be evicted. We must get a reference to it *before* it's evicted
      # so we can safely remove it from `valid_buffer` if it exists there.
      # The previous logic was flawed as it only checked the first element of
      # `valid_buffer`, leading to dangling pointers if a short episode was
      # evicted from the main buffer.
      evicted_episode = None
      if len(self.buffer) == self.capacity:
          evicted_episode = self.buffer[0]
              
      # Convert all lists to numpy arrays
      episode_dict = {}
      for key, values in self.current_episode.items():
          if key in ['obs', 'next_obs']:
                  # Store observations as uint8 to save memory
                  episode_dict[key] = np.array(values, dtype=np.uint8)
          elif key == 'actions':
              episode_dict[key] = np.array(values, dtype=np.int64)
          elif key == 'log_probs':
              episode_dict[key] = np.array(values, dtype=np.float32)
          elif key in ['rewards', 'state_reprs']:
              episode_dict[key] = np.array(values, dtype=np.float32)
          elif key in ['terminateds', 'truncateds']:
              episode_dict[key] = np.array(values, dtype=np.bool_)
      
      # This append operation is what actually evicts the oldest episode if capacity is reached.
      self.buffer.append(episode_dict)

      # Now, synchronize the valid_buffer.
      # 1. If an episode was evicted, remove it from valid_buffer if it was present.
      # We must iterate and check for identity (`is`) because `in` and `remove` use
      # `==`, which is ambiguous for dicts of numpy arrays and can cause crashes.
      if evicted_episode:
          for i, ep in enumerate(self.valid_buffer):
              if ep is evicted_episode:
                  del self.valid_buffer[i]
                  break # Found and removed, can stop searching.

      # 2. Add the newly committed episode to valid_buffer if it's long enough.
      if len(episode_dict['actions']) >= self.sequence_length:
          self.valid_buffer.append(episode_dict)

      self.reset_current_episode()

  def sample(self, batch_size: int, device: str) -> dict:
      """Samples a batch of transition sequences."""
      # --- OPTIMIZATION: Sample directly from the pre-filtered valid_buffer ---
      if not self.valid_buffer or batch_size == 0:
          return None

      # More efficient sampling: sample all episodes at once with replacement
      sampled_episodes = random.choices(self.valid_buffer, k=batch_size)

      batch_sequences = []
      for episode in sampled_episodes:
          episode_len = len(episode['actions'])
          start_idx = random.randint(0, episode_len - self.sequence_length)
          end_idx = start_idx + self.sequence_length
          
          sequence = {
              key: val[start_idx:end_idx] for key, val in episode.items()
          }
          batch_sequences.append(sequence)
      
      # Collate the batch of dictionaries into a single dictionary of tensors
      batch = {}
      for key in batch_sequences[0].keys():
          stacked = np.stack([seq[key] for seq in batch_sequences])
          tensor = torch.from_numpy(stacked).to(device)
          
          # Special handling for observations: convert uint8 to float32 and scale
          if key in ['obs', 'next_obs']:
              # The stored format is (B, S, H, W, C) as uint8.
              # We need to convert to (B, S, C, H, W) as float32 in [0, 1].
              tensor = tensor.permute(0, 1, 4, 2, 3).float() / 255.0

          batch[key] = tensor

      return batch

  def __len__(self) -> int:
      """Returns the number of episodes currently in the buffer."""
      return len(self.buffer)

  def clear(self):
      """Clears all episodes from the buffer."""
      self.buffer.clear()
      self.valid_buffer.clear()
      self.reset_current_episode()
