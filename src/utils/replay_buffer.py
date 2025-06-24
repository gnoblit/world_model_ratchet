import torch
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    A replay buffer that stores entire trajectories as dictionaries of NumPy arrays
    and samples sequences of transitions. This is more memory-efficient.
    """
    def __init__(self, cfg):
        self.capacity = cfg.capacity # Max number of *episodes* to store
        self.sequence_length = cfg.sequence_length
        self.buffer = deque(maxlen=self.capacity)
        self.reset_current_episode()

    def reset_current_episode(self):
        """Resets the temporary buffers for the current episode."""
        self.current_episode = {
            'obs': [], 'actions': [], 'rewards': [], 
            'next_obs': [], 'dones': []
        }

    def add(self, obs, action, reward, next_obs, terminated, truncated):
        """Adds a single transition to the current episode's temporary lists."""
        # Note: obs and next_obs from our wrapper are PyTorch tensors.
        # We convert them to numpy for efficient storage.
        self.current_episode['obs'].append(obs.cpu().numpy())
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['next_obs'].append(next_obs.cpu().numpy())
        self.current_episode['dones'].append(terminated or truncated)

        if terminated or truncated:
            self.commit_episode()

    def commit_episode(self):
        """Converts the current episode lists to NumPy arrays and stores them."""
        if not self.current_episode['actions']:
            return # Don't commit empty episodes

        # Convert all lists to numpy arrays
        episode_dict = {}
        for key, values in self.current_episode.items():
            dtype = np.float32 if key == 'rewards' else (np.bool_ if key == 'dones' else np.object_)
            if key in ['obs', 'next_obs']:
                 episode_dict[key] = np.array(values, dtype=np.float32) # Assuming observations are already scaled to [0,1]
            elif key == 'actions':
                episode_dict[key] = np.array(values, dtype=np.int64)
            else:
                episode_dict[key] = np.array(values, dtype=dtype)
        
        self.buffer.append(episode_dict)
        self.reset_current_episode()

    def sample(self, batch_size: int, device: str) -> dict:
        """Samples a batch of transition sequences."""
        # It's possible the buffer has episodes, but none are long enough.
        # First, get a list of episodes that are valid for sampling.
        valid_episodes = [ep for ep in self.buffer if len(ep['actions']) >= self.sequence_length]

        if not valid_episodes or batch_size == 0:
            return None

        batch_sequences = []
        # Use a while loop to ensure we collect exactly batch_size valid sequences
        while len(batch_sequences) < batch_size:
            episode = random.choice(valid_episodes)
            episode_len = len(episode['actions'])
            
            # This check is now redundant due to pre-filtering but is safe to keep
            if episode_len < self.sequence_length:
                continue

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
            batch[key] = torch.from_numpy(stacked).to(device)
            
        return batch

    def __len__(self) -> int:
        """Returns the number of episodes currently in the buffer."""
        return len(self.buffer)
