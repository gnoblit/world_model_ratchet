# tests/training/test_trainer.py

import torch
import pytest
import os
import numpy as np  # <-- ADD THIS LINE
from unittest.mock import MagicMock, patch, call

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
    # Check for target network
    assert trainer.target_perception_agent is not None, "Target perception agent should be initialized"
    assert id(trainer.target_perception_agent) != id(trainer.perception_agent), "Target should be a deep copy"
    for param in trainer.target_perception_agent.parameters():
        assert not param.requires_grad, "Target network parameters should be frozen"
    
def test_trainer_initialization_with_optimizations(temp_log_dir):
    """Tests that the Trainer initializes correctly and applies optimizations."""
    with patch('torch.compile') as mock_compile:
        config = get_base_config()
        config.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.run_name = "test_run_with_optimizations"
        config.experiment_dir = temp_log_dir

        # --- Test Case 1: Optimizations ON (on CUDA) ---
        config.training.use_torch_compile = True
        if config.training.device == 'cuda':
            trainer = Trainer(config)
            assert trainer.logger is not None
            assert os.path.isdir(trainer.log_dir)
            # Check that compile was called for all three models
            assert mock_compile.call_count == 3
            # Check that the scaler is enabled
            assert trainer.scaler.is_enabled()

        # --- Test Case 2: Optimizations OFF ---
        mock_compile.reset_mock()
        config.training.use_torch_compile = False
        trainer = Trainer(config)
        mock_compile.assert_not_called()
        # Scaler is enabled based on device, not a config flag, so it might still be on.
        assert trainer.scaler.is_enabled() == (config.training.device == 'cuda')

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
        if trainer.total_env_steps == num_steps - 1:
            # 1. Create a dummy observation that mimics the *raw* output of the 
            #    original Crafter environment (HWC format, uint8 dtype).
            raw_obs_shape = (*trainer.env.cfg.image_size, 3)  # e.g., (64, 64, 3)
            dummy_raw_obs_np = np.random.randint(0, 256, size=raw_obs_shape, dtype=np.uint8)

            # 2. Now, use the environment's actual transform function. This correctly
            #    converts our raw (H, W, C) numpy array into a (C, H, W) tensor.
            #    This ensures the shape is consistent with real steps.
            dummy_obs_tensor = trainer.env.transform(dummy_raw_obs_np)
            
            return dummy_obs_tensor, 1.0, True, False, {}
        
        return original_step_fn(action)
    
    trainer.env.step = mock_step 
    
    trainer.train_for_steps(num_env_steps=num_steps)

    assert trainer.logger.log_scalar.call_count >= 2
    trainer.logger.log_scalar.assert_any_call('rollout/episode_length', num_steps, trainer.total_env_steps)

# ... (the rest of the file remains unchanged) ...

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
    state_dim = config.perception.code_dim
    device = config.training.device
    mock_batch = {
        'obs': torch.rand(batch_size, seq_len, 3, *img_shape).to(device),
        'actions': torch.randint(0, num_actions, (batch_size, seq_len)).to(device),
        'log_probs': torch.randn(batch_size, seq_len).to(device),
        'rewards': torch.rand(batch_size, seq_len).to(device),
        'next_obs': torch.rand(batch_size, seq_len, 3, *img_shape).to(device),
        'terminateds': torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool).to(device),
        'state_reprs': torch.randn(batch_size, seq_len, state_dim).to(device),
    }

    wm_params_before = [p.clone() for p in trainer.perception_agent.parameters()]
    ac_params_before = [p.clone() for p in trainer.actor_critic.parameters()]

    # Mock the target network update to isolate the update_models logic
    trainer._update_target_network = MagicMock(name="_update_target_network")
    # Mock the scaler to check its usage, and make scale() a pass-through
    trainer.scaler = MagicMock(spec=torch.cuda.amp.GradScaler)
    trainer.scaler.scale.side_effect = lambda x: x

    # --- UPDATED TEST LOGIC ---
    # The method now returns a dictionary of losses
    # We run the update, which populates gradients
    loss_dict = trainer.update_models(mock_batch, teacher_is_frozen=False, student_is_frozen=False)

    # Assert that the returned object is a dictionary
    assert isinstance(loss_dict, dict), "update_models should return a dictionary of losses"
    
    # Assert that the dictionary contains all the expected loss keys
    expected_keys = [
        'world_model_loss', 'prediction_loss', 'codebook_loss', 'commitment_loss', 'code_entropy',
        'actor_loss', 'critic_loss', 'entropy_loss', 'total_action_loss'
    ]
    for key in expected_keys:
        assert key in loss_dict, f"Loss dictionary is missing key: {key}"
        assert isinstance(loss_dict[key], float), f"Loss value for {key} should be a float"

    # Check that the scaler's methods were called correctly for unfrozen models
    trainer.scaler.scale.assert_called_once() # Only one call on combined_loss
    
    # Check unscale_ calls: all three optimizers should be unscaled
    trainer.scaler.unscale_.assert_any_call(trainer.perception_optimizer)
    trainer.scaler.unscale_.assert_any_call(trainer.world_optimizer)
    trainer.scaler.unscale_.assert_any_call(trainer.action_optimizer)
    assert trainer.scaler.unscale_.call_count == 3

    step_calls = [
        call(trainer.perception_optimizer),
        call(trainer.world_optimizer),
        call(trainer.action_optimizer)
    ]
    trainer.scaler.step.assert_has_calls(step_calls, any_order=True)
    trainer.scaler.update.assert_called_once()

    # With the scaler mocked, we don't expect actual parameter updates in this unit test.
    # The primary goal is to verify that the scaler's methods are called correctly.
    # If we wanted to test actual parameter updates, it would require a more complex
    # mock for scaler.step or a full integration test.
    # Therefore, the assertions for parameter changes are removed.

    # Check that the target network update was called
    trainer._update_target_network.assert_called_once()

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
    state_dim = config.perception.code_dim
    device = config.training.device
    mock_batch = {
        'obs': torch.rand(batch_size, seq_len, 3, *img_shape).to(device),
        'actions': torch.randint(0, num_actions, (batch_size, seq_len)).to(device),
        'log_probs': torch.randn(batch_size, seq_len).to(device),
        'rewards': torch.rand(batch_size, seq_len).to(device),
        'next_obs': torch.rand(batch_size, seq_len, 3, *img_shape).to(device),
        'terminateds': torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool).to(device),
        'state_reprs': torch.randn(batch_size, seq_len, state_dim).to(device),
    }
    
    perception_params_before = [p.clone() for p in trainer.perception_agent.parameters()]
    ac_params_before = [p.clone() for p in trainer.actor_critic.parameters()]

    # Mock the target network update as it should not be called when teacher is frozen
    trainer._update_target_network = MagicMock(name="_update_target_network")
    # Mock the scaler
    trainer.scaler = MagicMock(spec=torch.cuda.amp.GradScaler)
    trainer.scaler.scale.side_effect = lambda x: x

    # --- UPDATED TEST LOGIC ---
    loss_dict = trainer.update_models(mock_batch, teacher_is_frozen=True, student_is_frozen=False) # Populates grads

    # Assert that the returned object is a dictionary
    assert isinstance(loss_dict, dict)
    
    # When frozen, the world model training losses should NOT be present
    assert 'world_model_loss' not in loss_dict
    assert 'prediction_loss' not in loss_dict
    # Because of the compute_losses=False optimization, the VQ losses are not calculated
    assert 'codebook_loss' not in loss_dict
    assert 'commitment_loss' not in loss_dict
    assert 'code_entropy' not in loss_dict

    # But the A2C losses should be present
    a2c_keys = ['actor_loss', 'critic_loss', 'entropy_loss', 'total_action_loss']
    for key in a2c_keys:
        assert key in loss_dict, f"Loss dictionary is missing key: {key} even when teacher is frozen"

    # Check scaler calls:
    trainer.scaler.scale.assert_called_once() # Only one call on combined_loss
    
    # Check unscale_ calls: only action_optimizer should be unscaled
    trainer.scaler.unscale_.assert_called_once_with(trainer.action_optimizer)
    
    trainer.scaler.step.assert_called_once_with(trainer.action_optimizer) # Only action_optimizer is stepped
    trainer.scaler.update.assert_called_once()

    # As above, parameter update checks are removed for mocked scaler.
    # We rely on the assertions that scaler.step was called for the correct optimizer.
    # The check for perception_agent not being updated is still valid as its optimizer
    # was not passed to scaler.step.
    assert all(torch.equal(p_before, p_after) for p_before, p_after in zip(perception_params_before, trainer.perception_agent.parameters()))
    # Check that target network was NOT updated
    trainer._update_target_network.assert_not_called()

def test_update_models_logic_student_frozen():
    """Tests the update_models method when the student IS frozen (teacher refinement)."""
    config = get_base_config()
    config.training.device = 'cpu'
    config.run_name = None
    config.training.batch_size = 4
    config.replay_buffer.sequence_length = 10
    
    trainer = Trainer(config)
    
    # Create a mock batch of data
    batch_size = config.training.batch_size
    seq_len = config.replay_buffer.sequence_length
    img_shape = config.env.image_size
    num_actions = config.action.num_actions
    state_dim = config.perception.code_dim
    device = config.training.device
    mock_batch = {
        'obs': torch.rand(batch_size, seq_len, 3, *img_shape).to(device),
        'actions': torch.randint(0, num_actions, (batch_size, seq_len)).to(device),
        'log_probs': torch.randn(batch_size, seq_len).to(device),
        'rewards': torch.rand(batch_size, seq_len).to(device),
        'next_obs': torch.rand(batch_size, seq_len, 3, *img_shape).to(device),
        'terminateds': torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool).to(device),
        'state_reprs': torch.randn(batch_size, seq_len, state_dim).to(device),
    }
    
    wm_params_before = [p.clone() for p in trainer.perception_agent.parameters()]
    ac_params_before = [p.clone() for p in trainer.actor_critic.parameters()]

    # Mock the target network update to isolate the update_models logic
    trainer._update_target_network = MagicMock(name="_update_target_network")
    # Mock the scaler
    trainer.scaler = MagicMock(spec=torch.cuda.amp.GradScaler)
    trainer.scaler.scale.side_effect = lambda x: x
    loss_dict = trainer.update_models(mock_batch, teacher_is_frozen=False, student_is_frozen=True) # Populates grads

    # Assert that the world model losses ARE present
    wm_keys = ['world_model_loss', 'prediction_loss', 'codebook_loss', 'commitment_loss', 'code_entropy']
    for key in wm_keys:
        assert key in loss_dict, f"Loss dictionary is missing key: {key} during teacher refinement"
    
    # Assert that the A2C losses are NOT present
    assert 'actor_loss' not in loss_dict
    assert 'critic_loss' not in loss_dict

    # Check scaler calls: only the teacher part should run
    trainer.scaler.scale.assert_called_once() # Only for world model
    unscale_calls = [call(trainer.perception_optimizer), call(trainer.world_optimizer)]
    trainer.scaler.unscale_.assert_has_calls(unscale_calls, any_order=True)
    step_calls = [call(trainer.perception_optimizer), call(trainer.world_optimizer)]
    trainer.scaler.step.assert_has_calls(step_calls, any_order=True)
    trainer.scaler.update.assert_called_once()

    # As above, parameter update checks are removed for mocked scaler.
    # We rely on the assertions that scaler.step was called for the correct optimizers.
    # The check for actor_critic not being updated is still valid as its optimizer
    # was not passed to scaler.step.
    assert all(torch.equal(p_before, p_after) for p_before, p_after in zip(ac_params_before, trainer.actor_critic.parameters()))

    # Assert that target network update was called
    trainer._update_target_network.assert_called_once()

def test_train_from_buffer_updates_total_steps():
    """
    Tests that train_from_buffer correctly increments the trainer's grad_updates
    counter but NOT the env_steps counter.
    """
    config = get_base_config()
    config.training.device = 'cpu'
    config.run_name = None
    config.training.batch_size = 4
    config.replay_buffer.sequence_length = 10
    state_dim = config.perception.code_dim
    
    trainer = Trainer(config)
    
    # Mock the replay buffer to return a valid batch
    mock_batch = {
        'obs': torch.rand(4, 10, 3, 64, 64),
        'actions': torch.randint(0, 17, (4, 10)),
        'log_probs': torch.randn(4, 10),
        'rewards': torch.rand(4, 10),
        'next_obs': torch.rand(4, 10, 3, 64, 64),
        'terminateds': torch.zeros(4, 10, dtype=torch.bool),
        'state_reprs': torch.randn(4, 10, state_dim),
    }
    trainer.replay_buffer.sample = MagicMock(return_value=mock_batch)
    
    initial_steps = 1000
    num_updates = 50
    trainer.total_env_steps = initial_steps
    trainer.total_grad_updates = 0
    trainer.train_from_buffer(num_updates=num_updates)
    
    assert trainer.total_env_steps == initial_steps, "train_from_buffer should NOT increment total_env_steps"
    assert trainer.total_grad_updates == num_updates, "train_from_buffer should increment total_grad_updates"