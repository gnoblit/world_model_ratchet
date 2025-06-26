import pytest
import torch
from configs.base_config import get_base_config
from environment.crafter_env import CrafterEnvWrapper

def test_env_initialization():
    """Tests that the CrafterEnvWrapper can be initialized."""
    config = get_base_config().env
    try:
        env = CrafterEnvWrapper(config)
        assert env is not None
        env.close()
    except Exception as e:
        pytest.fail(f"CrafterEnvWrapper initialization failed: {e}")

def test_env_reset_reproducibility():
    """Tests that resetting the environment with the same seed produces the same observation."""
    config = get_base_config().env
    env = CrafterEnvWrapper(config)
    
    # Reset with a specific seed and get the first observation
    obs1, _ = env.reset(seed=42)
    
    # Take a few random steps to change the state
    for _ in range(5):
        action = env.action_space.sample()
        env.step(action)
        
    # Reset again with the same seed
    obs2, _ = env.reset(seed=42)
    
    # The observations should be identical
    assert torch.equal(obs1, obs2), "Resetting with the same seed should yield the same observation."
    
    # Reset with a different seed
    obs3, _ = env.reset(seed=99)
    
    # This observation should be different
    assert not torch.equal(obs1, obs3), "Resetting with a different seed should yield a different observation."
    
    env.close()

def test_env_step_output():
    """Tests the output of the step function."""
    config = get_base_config().env
    env = CrafterEnvWrapper(config)
    env.reset(seed=42)
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check types and shapes
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (3, *config.image_size)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    env.close()