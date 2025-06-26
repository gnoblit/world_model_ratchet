import crafter
import gymnasium as gym
import numpy as np
from torchvision import transforms

class CrafterEnvWrapper(gym.Env):
    """A wrapper for the Crafter environment that conforms to the Gymnasium API"""

    def __init__(self, cfg):
        """
        Initializes the Crafter environment wrapper.
        
        Args:
            cfg (EnvConfig): The environment configuration object.
        """
        self.cfg = cfg # Store config for re-initialization on reset
        # Original Crafter env has a distinct API
        self._env = crafter.Env(size=cfg.image_size, seed=cfg.seed)

        # Define observation and action spaces according to Gymansium standards
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*cfg.image_size, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(self._env.action_space.n)

        # Define PyTorch transform to convert observations
        self.transform = transforms.ToTensor()

    def _preprocess_obs(self, obs):
        """
        Preprocesses the observation from Crafter.
        Convers from HWC (Height, Width, Channel) to CHW (Channel, Heigh, Width)
        Scales to [0, 1] as a PyTorch tensor.

        transforms.ToTensor does this automatically.
        """
        tensor_obs = self.transform(obs.copy())
        return tensor_obs
    
    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (int): The action to take.
        
        Returns:
        A tuple (observation, reward, terminated, truncated, info).
        """
        obs, reward, done, info = self._env.step(action)
        terminated = done
        truncated = False # No time limit truncation in Crafter

        return self._preprocess_obs(obs), reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """
        Resets the environment.

        Returns:
        A tuple (observation, info).
        """
        if seed is not None:
            # The Crafter environment must be re-initialized to be seeded,
            # as it does not support seeding on reset. This is the most
            # reliable way to ensure reproducibility.
            self._env = crafter.Env(size=self.cfg.image_size, seed=seed)
        obs = self._env.reset()
        info = {} # Info dict is empty on reset

        return self._preprocess_obs(obs), info
    
    def render(self, mode='human'):
        """Renders the environment."""

        return self._env.render()
    
    def close(self):
        """Crafter lacks a close so just passes."""
        pass
