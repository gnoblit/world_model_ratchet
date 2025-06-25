import torch
import pytest
import os
from unittest.mock import MagicMock

from configs.base_config import get_base_config
from training.trainer import Trainer

# The 'temp_log_dir' fixture is now automatically available from tests/conftest.py
# We do not need to define or import it here.

def test_trainer_initialization_no_logger():
    """Tests that the Trainer initializes without a logger if run_name is None."""
    config = get_base_config()
    config.training.device = 'cpu'
    config.run_name = None # Explicitly disable logger creation
    
    trainer = Trainer(config)
    assert trainer is not None, "Trainer should not be None"
    assert trainer.logger is None, "Logger should be None when run_name is not set"
    assert trainer.log_dir is None, "Log directory should be None when run_name is not set"
    
def test_trainer_initialization_with_logger(temp_log_dir): # Use the new fixture name
    """Tests that the Trainer initializes correctly WITH a logger."""
    config = get_base_config()
    config.training.device = 'cpu'
    config.run_name = "test_run_with_logger"
    config.experiment_dir = temp_log_dir # Use the path provided by the fixture
    
    trainer = Trainer(config)
    assert trainer is not None, "Trainer should not be None"
    assert trainer.logger is not None, "Logger should be initialized"
    assert os.path.isdir(trainer.log_dir), "Log directory should have been created"

def test_trainer_logging_on_episode_end(temp_log_dir): # Use the new fixture name
    """Tests that episode stats are logged when an episode terminates."""
    config = get_base_config()
    config.training.device = 'cpu'
    config.run_name = "test_logging_run"
    config.experiment_dir = temp_log_dir # Use the path provided by the fixture
    num_steps = 5
    
    trainer = Trainer(config)
    
    # We mock the *real* logger object that was created
    trainer.logger = MagicMock()
    
    original_step_fn = trainer.env.step
    def mock_step(action):
        if trainer.total_steps == num_steps - 1:
            dummy_obs_np = trainer.env.observation_space.sample()
            dummy_obs_tensor = trainer.env.transform(dummy_obs_np.copy())
            return dummy_obs_tensor, 1.0, True, False, {}
        return original_step_fn(action)
    trainer.env.step = mock_step 
    
    trainer.train_for_steps(num_steps=num_steps)

    assert trainer.logger.log_scalar.call_count >= 2
    trainer.logger.log_scalar.assert_any_call('rollout/episode_length', num_steps, num_steps)

def test_update_models_logic_unfrozen():
    """Tests the update_models method when the teacher is NOT frozen."""
    config = get_base_config()
    config.training.device = 'cpu'
    config.run_name = None
    config.training.batch_size = 4
    config.replay_buffer.sequence_length = 10
    
    trainer = Trainer(config)
    
    # ... (mock_batch setup is the same)
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

    wm_params_before = [p.clone() for p in trainer.perception_agent.parameters()]
    ac_params_before = [p.clone() for p in trainer.actor_critic.parameters()]
    
    # --- UPDATED TEST LOGIC ---
    # The method now returns a dictionary of losses
    loss_dict = trainer.update_models(mock_batch, teacher_is_frozen=False)

    # Assert that the returned object is a dictionary
    assert isinstance(loss_dict, dict), "update_models should return a dictionary of losses"
    
    # Assert that the dictionary contains all the expected loss keys
    expected_keys = [
        'world_model_loss', 'prediction_loss', 'commitment_loss',
        'actor_loss', 'critic_loss', 'entropy_loss', 'total_action_loss'
    ]
    for key in expected_keys:
        assert key in loss_dict, f"Loss dictionary is missing key: {key}"
        assert isinstance(loss_dict[key], float), f"Loss value for {key} should be a float"
    # --- END OF UPDATED LOGIC ---

    wm_params_after = list(trainer.perception_agent.parameters())
    assert any(not torch.equal(p_before, p_after) for p_before, p_after in zip(wm_params_before, wm_params_after))
    ac_params_after = list(trainer.actor_critic.parameters())
    assert any(not torch.equal(p_before, p_after) for p_before, p_after in zip(ac_params_before, ac_params_after))

def test_update_models_logic_frozen():
    """Tests the update_models method when the teacher IS frozen."""
    config = get_base_config()
    config.training.device = 'cpu'
    config.run_name = None
    config.training.batch_size = 4
    config.replay_buffer.sequence_length = 10
    
    trainer = Trainer(config)
    
    # ... (mock_batch setup is the same)
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
    
    wm_params_before = [p.clone() for p in trainer.perception_agent.parameters()]
    ac_params_before = [p.clone() for p in trainer.actor_critic.parameters()]

    # --- UPDATED TEST LOGIC ---
    loss_dict = trainer.update_models(mock_batch, teacher_is_frozen=True)

    # Assert that the returned object is a dictionary
    assert isinstance(loss_dict, dict)
    
    # When frozen, the world model losses should NOT be present
    assert 'world_model_loss' not in loss_dict
    assert 'prediction_loss' not in loss_dict
    assert 'commitment_loss' not in loss_dict
    
    # But the A2C losses should be present
    a2c_keys = ['actor_loss', 'critic_loss', 'entropy_loss', 'total_action_loss']
    for key in a2c_keys:
        assert key in loss_dict, f"Loss dictionary is missing key: {key} even when teacher is frozen"
    # --- END OF UPDATED LOGIC ---

    wm_params_after = list(trainer.perception_agent.parameters())
    assert all(torch.equal(p_before, p_after) for p_before, p_after in zip(wm_params_before, wm_params_after))
    ac_params_after = list(trainer.actor_critic.parameters())
    assert any(not torch.equal(p_before, p_after) for p_before, p_after in zip(ac_params_before, ac_params_after))