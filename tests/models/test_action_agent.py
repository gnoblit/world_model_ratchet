import torch
import pytest  
from configs.base_config import get_base_config
from models.action import ActionAgent

def test_action_agent_initialization():
    """Tests that the ActionAgent can be initialized."""
    config = get_base_config()
    try:
        agent = ActionAgent(
            state_dim=config.perception.code_dim,
            num_actions=config.action.num_actions,
        )
        assert agent is not None, "Agent should exist after init."
    except Exception as e:
        pytest.fail(f"ActionAgent initialization failed with error: {e}")

def test_action_agent_forward_pass():
    """Tests the forward pass for the correct output shape."""
    config = get_base_config()
    agent = ActionAgent(
        state_dim=config.perception.code_dim,
        num_actions=config.action.num_actions,
    )
    batch_size = 4
    dummy_state_batch = torch.randn(batch_size, config.perception.code_dim)

    action_logits = agent.forward(dummy_state_batch)

    expected_shape = (batch_size, config.action.num_actions)

    assert action_logits.shape == expected_shape, \
        f"Expected shape {expected_shape}, but got {action_logits.shape}"

def test_action_agent_get_action():
    """Tests the get_action helper method for correct types and value ranges."""
    config = get_base_config()
    agent = ActionAgent(
        state_dim=config.perception.code_dim, 
        num_actions=config.action.num_actions
    )
    
    dummy_single_state = torch.randn(1, config.perception.code_dim)

    # Test stochastic action
    action, log_prob = agent.get_action(dummy_single_state, deterministic=False)
    assert isinstance(action, int), "Stochastic action should be an integer."
    assert 0 <= action < config.action.num_actions, "Stochastic action out of range."
    assert isinstance(log_prob, torch.Tensor), "Log prob should be a tensor."

    # Test deterministic action
    action_det, _ = agent.get_action(dummy_single_state, deterministic=True)
    assert isinstance(action_det, int), "Deterministic action should be an integer."
    assert 0 <= action_det < config.action.num_actions, "Deterministic action out of range."