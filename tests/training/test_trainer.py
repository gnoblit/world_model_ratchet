import torch
import pytest
import copy

from configs.base_config import get_base_config
from training.trainer import Trainer

def test_trainer_initialization():
    """Tests that the Trainer and all its components can be initialized."""
    try:
        config = get_base_config()
        config.training.device = 'cpu'
        trainer = Trainer(config)
        assert trainer is not None, "Trainer should not be None"
    except Exception as e:
        pytest.fail(f"Trainer initialization failed: {e}")

def test_update_models_logic_unfrozen():
    """
    Tests the update_models method when the teacher is NOT frozen.
    Checks that all models are updated.
    """
    config = get_base_config()
    config.training.device = 'cpu'
    config.training.batch_size = 4
    config.replay_buffer.sequence_length = 10
    
    trainer = Trainer(config)
    
    # Create a mock batch of data
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

    # Store original parameters
    wm_params_before = [p.clone() for p in trainer.perception_agent.parameters()]
    ac_params_before = [p.clone() for p in trainer.actor_critic.parameters()]
    
    # Call update_models with teacher_is_frozen=False
    try:
        trainer.update_models(mock_batch, teacher_is_frozen=False) # <--- THE FIX
    except Exception as e:
        pytest.fail(f"trainer.update_models(teacher_is_frozen=False) failed with an error: {e}")

    # Assert that ALL model weights have been updated
    wm_params_after = list(trainer.perception_agent.parameters())
    assert any(not torch.equal(p_before, p_after) for p_before, p_after in zip(wm_params_before, wm_params_after)), \
        "World model/perception parameters should be updated when teacher is not frozen."

    ac_params_after = list(trainer.actor_critic.parameters())
    assert any(not torch.equal(p_before, p_after) for p_before, p_after in zip(ac_params_before, ac_params_after)), \
        "Actor-Critic parameters should be updated."

def test_update_models_logic_frozen():
    """
    Tests the update_models method when the teacher IS frozen.
    Checks that only the student (Actor-Critic) is updated.
    """
    config = get_base_config()
    config.training.device = 'cpu'
    config.training.batch_size = 4
    config.replay_buffer.sequence_length = 10
    
    trainer = Trainer(config)
    
    # Create a mock batch (same as before)
    # ... (code to create mock_batch is identical)
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
    
    # Store original parameters
    wm_params_before = [p.clone() for p in trainer.perception_agent.parameters()]
    ac_params_before = [p.clone() for p in trainer.actor_critic.parameters()]

    # Call update_models with teacher_is_frozen=True
    try:
        trainer.update_models(mock_batch, teacher_is_frozen=True)
    except Exception as e:
        pytest.fail(f"trainer.update_models(teacher_is_frozen=True) failed with an error: {e}")

    # Assert that ONLY the Actor-Critic weights have changed
    wm_params_after = list(trainer.perception_agent.parameters())
    assert all(torch.equal(p_before, p_after) for p_before, p_after in zip(wm_params_before, wm_params_after)), \
        "World model/perception parameters should NOT be updated when teacher is frozen."

    ac_params_after = list(trainer.actor_critic.parameters())
    assert any(not torch.equal(p_before, p_after) for p_before, p_after in zip(ac_params_before, ac_params_after)), \
        "Actor-Critic parameters should still be updated when teacher is frozen."