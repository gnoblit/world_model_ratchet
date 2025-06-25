import torch
import pytest
from configs.base_config import get_base_config
from models.actor_critic import ActorCritic

def test_actor_critic_initialization():
    config = get_base_config()
    try:
        model = ActorCritic(
            state_dim=config.perception.code_dim,
            num_actions=config.action.num_actions,
            cfg=config.action,
        )
        assert model is not None
        assert model.actor is not None
        assert model.critic is not None
    except Exception as e:
        pytest.fail(f"ActorCritic initialization failed: {e}")

def test_get_action_and_get_value():
    """Tests the helper methods for both single and batch inputs."""
    config = get_base_config()
    model = ActorCritic(
        state_dim=config.perception.code_dim,
        num_actions=config.action.num_actions,
        cfg=config.action,
    )

    # --- Test with single input (batch size 1) ---
    dummy_single_state = torch.randn(1, config.perception.code_dim)

    action, log_prob = model.get_action(dummy_single_state)
    assert isinstance(action, int)
    assert 0 <= action < config.action.num_actions
    # This assertion will now pass
    assert log_prob.shape == (), f"Expected scalar log_prob, but got shape {log_prob.shape}"

    value = model.get_value(dummy_single_state)
    assert value.shape == (1, 1)

    # --- Test with batch input ---
    batch_size = 4
    dummy_batch_state = torch.randn(batch_size, config.perception.code_dim)

    actions_batch, log_probs_batch = model.get_action(dummy_batch_state)
    assert isinstance(actions_batch, torch.Tensor)
    assert actions_batch.shape == (batch_size,)
    assert log_probs_batch.shape == (batch_size,)