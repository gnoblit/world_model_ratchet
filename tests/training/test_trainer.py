import torch
import pytest
import copy

from configs.base_config import get_base_config
from training.trainer import Trainer

def test_trainer_initialization():
    """Tests that the Trainer and all its components can be initialized."""
    try:
        config = get_base_config()
        # Force CPU for testing to avoid GPU dependencies in CI/local envs
        config.training.device = 'cpu'
        trainer = Trainer(config)
        assert trainer is not None, "Trainer should not be None"
        assert trainer.device == 'cpu'
    except Exception as e:
        pytest.fail(f"Trainer initialization failed: {e}")

def test_update_models_logic():
    """
    Tests the core logic of the update_models method.
    Checks that it runs without errors and that model weights are updated.
    """
    # --- 1. Setup ---
    config = get_base_config()
    config.training.device = 'cpu'
    
    # Use a smaller batch and sequence length for a faster test
    config.training.batch_size = 4
    config.replay_buffer.sequence_length = 10
    
    trainer = Trainer(config)
    
    # --- 2. Create a mock batch of data ---
    batch_size = config.training.batch_size
    seq_len = config.replay_buffer.sequence_length
    img_shape = config.env.image_size
    num_actions = config.action.num_actions
    device = config.training.device

    mock_batch = {
        'obs': torch.rand(batch_size, seq_len, 3, *img_shape).to(device),
        'actions': torch.randint(0, num_actions, (batch_size, seq_len)).to(device),
        'rewards': torch.rand(batch_size, seq_len).to(device),
        'next_obs': torch.rand(batch_size, seq_len, 3, *img_shape).to(device),
        'dones': torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool).to(device),
    }

    # --- 3. Run the update and verify it doesn't crash ---
    try:
        # Store original parameters to check if they change
        wm_params_before = [p.clone() for p in trainer.perception_agent.parameters()]
        ac_params_before = [p.clone() for p in trainer.actor_critic.parameters()]
        
        trainer.update_models(mock_batch)
    except Exception as e:
        pytest.fail(f"trainer.update_models() failed with an error: {e}")

    # --- 4. Assert that the model weights have been updated ---
    # We check that at least one parameter in each model has changed.
    
    # Check World Model / Perception Agent
    wm_params_after = list(trainer.perception_agent.parameters())
    assert any(not torch.equal(p_before, p_after) for p_before, p_after in zip(wm_params_before, wm_params_after)), \
        "World model/perception parameters were not updated after the step."

    # Check Actor-Critic
    ac_params_after = list(trainer.actor_critic.parameters())
    assert any(not torch.equal(p_before, p_after) for p_before, p_after in zip(ac_params_before, ac_params_after)), \
        "Actor-Critic parameters were not updated after the step."