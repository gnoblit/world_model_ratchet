import crafter
import gymnasium as gym
import numpy as np
import torch
from torchvision import transforms
from typing import Tuple, Any, Dict

from configs.base_config import EnvConfig

class CrafterEnvWrapper(gym.Env):
    """
    A Gymnasium-compliant wrapper for the Crafter environment.

    This wrapper handles:
    1.  Conversion of observations from HWC NumPy arrays to CHW PyTorch tensors.
    2.  Correction of the observation space to reflect the preprocessed output.
    3.  A robust but slow seeding mechanism, necessitated by the Crafter API.
    """
    # The output of step() is a tensor, not a numpy array.
    # The space should reflect what the wrapper *returns*.
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, cfg: EnvConfig):
        """
        Initializes the Crafter environment wrapper.
        
        Args:
            cfg (EnvConfig): The environment configuration object.
        """
        super().__init__()
        self.cfg = cfg
        
        # This is the standard way to handle seeding in Gymnasium.
        # We create a random number generator that will be used to seed
        # the environment on resets.
        self.np_random, _ = gym.utils.seeding.np_random(cfg.seed)

        # Initialize the underlying env. We'll re-create it on reset.
        self._env = self._create_env(self.cfg.seed)

        # 1. IMPROVEMENT: Correct Observation Space
        # The wrapper returns a normalized PyTorch tensor in CHW format.
        # The observation space MUST match the format of the returned observation.
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, *cfg.image_size), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self._env.action_space.n)

        # Store the transform to avoid re-creating it.
        self.transform = transforms.ToTensor()

    def _create_env(self, seed: int) -> crafter.Env:
        """Helper to create a new instance of the Crafter environment."""
        return crafter.Env(size=self.cfg.image_size, seed=seed)

    def _preprocess_obs(self, obs: np.ndarray) -> torch.Tensor:
        """
        Preprocesses the HWC NumPy observation to a CHW float tensor.
        
        2. IMPROVEMENT: Efficiency
        - Removed the unnecessary .copy(). transforms.ToTensor() does not
          modify the input NumPy array in-place; it creates a new tensor.
          The copy was redundant and wasted memory/CPU cycles.
        """
        return self.transform(obs)
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        """
        obs, reward, done, info = self._env.step(action)
        terminated = done
        # Crafter's 'done' flag signifies the end of an episode, not a time-limit.
        truncated = False 

        return self._preprocess_obs(obs), reward, terminated, truncated, info
    
    def reset(self, *, seed: int = None, options: Dict = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Resets the environment.

        3. IMPROVEMENT: Robust Seeding
        - The original `crafter.Env` lacks a `seed()` method and must be
          re-initialized to be re-seeded. This is a known performance bottleneck.
        - This implementation follows the standard Gymnasium pattern. If a seed
          is passed to reset(), we re-create the environment with that seed.
        - If no seed is passed, we draw a new seed from our internal random
          number generator (`self.np_random`) to ensure the sequence of
          episodes is deterministic if the wrapper itself was seeded at init.
        - This makes the wrapper's behavior predictable and standard, even if
          the underlying re-instantiation is slow.
        """
        super().reset(seed=seed, options=options)
        
        # Use the provided seed, or draw a new one from our RNG for reproducibility.
        # The cast to int is important as np_random can return a np.uint32.
        reset_seed = seed if seed is not None else int(self.np_random.integers(2**31 - 1))
        
        # Re-create the environment. This is slow but required by Crafter.
        self._env = self._create_env(seed=reset_seed)

        # The crafter.Env constructor calls reset() internally. However, as shown
        # by the test failures, calling render() immediately after is not safe
        # as the player object may not be initialized. A subsequent call to
        # reset() is required to get the initial observation and guarantee state.
        obs = self._env.reset()
        info = {} 

        return self._preprocess_obs(obs), info
    
    def render(self) -> np.ndarray:
        """Renders the environment."""
        return self._env.render()
    
    def close(self):
        """Closes the environment."""
        # The underlying Crafter env does not have a close method.
        pass