import torch
from src.configs.base_config import get_base_config
from src.models.action import ActionAgent

def test_action_agent():
    """
    Performs a set of checks to ensure the ActionAgent module is working correctly.
    """
    print("--- Running Test for ActionAgent ---")

    # 1. Load configuration to get model parameters
    config = get_base_config()
    action_config = config.action
    perception_config = config.perception
    
    # The state_dim for the ActionAgent must match the output of the PerceptionAgent
    state_dim = perception_config.code_dim 
    num_actions = action_config.num_actions
    
    print(f"Initializing ActionAgent with state_dim={state_dim}, num_actions={num_actions}")
    
    # 2. Instantiate the ActionAgent
    try:
        agent = ActionAgent(state_dim=state_dim, num_actions=num_actions)
        print("[SUCCESS] ActionAgent initialized successfully.")
    except Exception as e:
        print(f"[FAILURE] Could not initialize ActionAgent. Error: {e}")
        return

    # 3. Test the forward pass (for training)
    print("\n--- Testing forward() pass ---")
    try:
        # Create a dummy batch of state representations
        batch_size = 4
        dummy_state_batch = torch.randn(batch_size, state_dim)
        print(f"Input batch shape: {dummy_state_batch.shape}")
        
        # Pass the batch through the agent
        action_logits = agent.forward(dummy_state_batch)
        print(f"Output logits shape: {action_logits.shape}")

        # Check if the output shape is correct
        expected_shape = (batch_size, num_actions)
        assert action_logits.shape == expected_shape, "Output shape is incorrect!"
        print("[SUCCESS] forward() pass produced the correct output shape.")
        
    except Exception as e:
        print(f"[FAILURE] forward() pass failed. Error: {e}")
        return

    # 4. Test the get_action() method (for inference/gameplay)
    print("\n--- Testing get_action() method ---")
    try:
        # Create a single dummy state representation (needs a batch dimension)
        dummy_single_state = torch.randn(1, state_dim)
        print(f"Input state shape: {dummy_single_state.shape}")

        # --- Test stochastic action sampling ---
        action, log_prob = agent.get_action(dummy_single_state, deterministic=False)
        print(f"Sampled Action (stochastic): {action}")
        print(f"Log Probability: {log_prob.item():.4f}")

        # Check types and values
        assert isinstance(action, int), "Action should be an integer."
        assert 0 <= action < num_actions, "Action is out of valid range."
        assert isinstance(log_prob, torch.Tensor), "Log probability should be a tensor."
        print("[SUCCESS] Stochastic get_action() produced correct types and values.")
        
        # --- Test deterministic action selection ---
        action_det, log_prob_det = agent.get_action(dummy_single_state, deterministic=True)
        print(f"\nSampled Action (deterministic): {action_det}")
        
        assert isinstance(action_det, int), "Deterministic action should be an integer."
        print("[SUCCESS] Deterministic get_action() produced correct types.")

    except Exception as e:
        print(f"[FAILURE] get_action() method failed. Error: {e}")
        return
        
    print("\n--- ActionAgent Test Complete: All checks passed! ---")


if __name__ == "__main__":
    test_action_agent()