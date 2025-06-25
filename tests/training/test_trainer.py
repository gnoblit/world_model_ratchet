import torch
import pytest
from unittest.mock import MagicMock, call # Import MagicMock

from configs.base_config import get_base_config
from training.trainer import Trainer

def test_trainer_initialization():
    """Tests that the Trainer and all its components can be initialized."""
    try:
        config = get_base_config()
        config.training.device = 'cpu'
        trainer = Trainer(config)
        assert trainer is not None, "Trainer should not be None"
        assert trainer.logger is not None 
    except Exception as e:
        pytest.fail(f"Trainer initialization failed: {e}")

def test_trainer_logging_on_episode_end():
    """Tests that episode stats are logged when an episode terminates."""
    try:
        config = get_base_config()
        config.training.device = 'cpu'
        num_steps = 5
        trainer = Trainer(config)
    except Exception as e:
        pytest.fail(f"Test setup failed during Trainer initialization: {e}")

    # Mock the logger to spy on its calls
    trainer.logger = MagicMock()
    
    # --- Simulate a short episode that terminates ---
    # Manually set terminated to True on the last step
    original_step_fn = trainer.env.step
    def mock_step(action):
        if trainer.total_steps == num_steps - 1:
            # This now returns a (H, W, C) numpy array
            dummy_obs_np = trainer.env.observation_space.sample()
            # ToTensor converts it to (C, H, W) tensor, which is what the real step fn returns
            dummy_obs_tensor = trainer.env.transform(dummy_obs_np.copy())
            return dummy_obs_tensor, 1.0, True, False, {}
        return original_step_fn(action)
        
    # Monkey-patch the environment's step function
    trainer.env.step = mock_step 
    
    # --- Run the training loop ---
    try:
        trainer.train_for_steps(num_steps=num_steps)
    except Exception as e:
        pytest.fail(f"trainer.train_for_steps() raised an unexpected exception: {e}")

    # --- Assertions ---
    try:
        # Check that the logger was called at least twice (for reward and length)
        assert trainer.logger.log_scalar.call_count >= 2, \
            f"Expected at least 2 log calls, but got {trainer.logger.log_scalar.call_count}"
        
        # The total reward will be the sum of rewards from each step.
        # Since our mock returns 1.0 on the final step and the real env returns
        # a small float on others, it's hard to predict the exact total.
        # It's better to check that the call was made with the correct tag and step.
        # A more advanced mock could control the reward for every step.

        # Let's check the final call more robustly.
        # After 5 steps, the final episode length is 5 and total_steps is 5.
        # The final reward is the sum of rewards. Let's assume the mock's 1.0 dominates.
        
        # Use assert_any_call which is more flexible than checking the entire call list.
        trainer.logger.log_scalar.assert_any_call(
            'rollout/episode_length', 
            num_steps,  # The episode length should be 5
            num_steps   # The global step should be 5
        )
        
        # We can also check the reward call, but the value is less predictable.
        # Let's find the call for 'rollout/episode_reward' and check its arguments.
        reward_logged = False
        for c in trainer.logger.log_scalar.call_args_list:
            tag, value, step = c.args
            if tag == 'rollout/episode_reward' and step == num_steps:
                reward_logged = True
                # We can assert the value is a float, for instance
                assert isinstance(value, float)
                break
        assert reward_logged, "Episode reward was not logged on termination."

    except AssertionError as e:
        pytest.fail(f"Assertion failed during log verification: {e}")

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