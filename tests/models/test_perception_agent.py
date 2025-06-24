import torch
import pytest
from configs.base_config import get_base_config
from models.perception import PerceptionAgent, VisionEncoder, SharedCodebook

# --- Test Individual Components First ---

def test_vision_encoder():
    """Tests the VisionEncoder for correct output shape."""
    config = get_base_config().perception
    encoder = VisionEncoder(feature_dim=config.feature_dim)

    batch_size = 4
    # Use the image size from the main config's env settings
    image_size = get_base_config().env.image_size 
    dummy_obs_batch = torch.randn(batch_size, 3, *image_size)

    features = encoder(dummy_obs_batch)

    expected_shape = (batch_size, config.feature_dim)
    assert features.shape == expected_shape, \
        f"VisionEncoder expected shape {expected_shape}, but got {features.shape}"

def test_shared_codebook():
    """Tests the SharedCodebook for correct output shape and logic."""
    config = get_base_config().perception
    codebook = SharedCodebook(num_codes=config.num_codes, code_dim=config.code_dim)

    batch_size = 4
    # The input dimension to the codebook must match its code_dim
    dummy_features = torch.randn(batch_size, config.code_dim)

    representation = codebook(dummy_features)

    expected_shape = (batch_size, config.code_dim)
    assert representation.shape == expected_shape, \
        f"SharedCodebook expected shape {expected_shape}, but got {representation.shape}"

    # Test that it fails with incorrect input dimensions
    with pytest.raises(ValueError):
        wrong_dim_features = torch.randn(batch_size, config.code_dim + 1)
        codebook(wrong_dim_features)

# --- Test the Assembled Agent ---

def test_perception_agent_initialization():
    """Tests that the PerceptionAgent can be initialized."""
    config = get_base_config()
    try:
        agent = PerceptionAgent(config.perception)
        assert agent is not None, "Agent should not be None after initialization."
    except Exception as e:
        pytest.fail(f"PerceptionAgent initialization failed with error: {e}")

    # Test that initialization fails if dimensions don't match
    with pytest.raises(ValueError):
        faulty_config = config.perception
        # Deliberately create a mismatch
        faulty_config.feature_dim = 128
        faulty_config.code_dim = 256
        PerceptionAgent(faulty_config)

def test_perception_agent_full_forward_pass():
    """Tests the full forward pass from pixels to final representation."""
    config = get_base_config()
    # Reset config to be safe after the faulty one in the previous test
    perception_config = get_base_config().perception
    agent = PerceptionAgent(perception_config)

    batch_size = 4
    image_size = config.env.image_size
    dummy_obs_batch = torch.randn(batch_size, 3, *image_size)
    
    representation = agent(dummy_obs_batch)

    expected_shape = (batch_size, perception_config.code_dim)
    assert representation.shape == expected_shape, \
        f"PerceptionAgent expected shape {expected_shape}, but got {representation.shape}"