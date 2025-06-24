import torch
import pytest
from configs.base_config import get_base_config
from models.world_model import WorldModel

def test_world_model_initialization_and_forward_pass():
    """Tests the WorldModel for correct initialization and output shape."""
    config = get_base_config()
    
    # Get the specific configs needed
    wm_config = config.world_model
    state_dim = config.perception.code_dim
    num_actions = config.action.num_actions

    # Instantiate the model correctly, passing the config object
    try:
        world_model = WorldModel(
            state_dim=state_dim, 
            num_actions=num_actions,
            cfg=wm_config  # Pass the config object
        )
        assert world_model is not None, "WorldModel should not be None"
    except Exception as e:
        pytest.fail(f"WorldModel initialization failed: {e}")

    batch_size = 4
    
    # Create dummy inputs
    dummy_z_t = torch.randn(batch_size, state_dim)
    dummy_a_t = torch.randint(0, num_actions, (batch_size,)) 

    # Get the prediction
    predicted_z_hat_t1 = world_model(dummy_z_t, dummy_a_t)

    # Assert the output shape is correct
    expected_shape = (batch_size, state_dim)
    assert predicted_z_hat_t1.shape == expected_shape, \
        f"WorldModel expected shape {expected_shape}, but got {predicted_z_hat_t1.shape}"