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
    image_size = get_base_config().env.image_size 
    dummy_obs_batch = torch.randn(batch_size, 3, *image_size)
    features = encoder(dummy_obs_batch)
    expected_shape = (batch_size, config.feature_dim)
    assert features.shape == expected_shape, \
        f"VisionEncoder expected shape {expected_shape}, but got {features.shape}"

def test_shared_codebook():
    """Tests the SharedCodebook for correct return types and shapes."""
    config = get_base_config().perception
    codebook = SharedCodebook(num_codes=config.num_codes, code_dim=config.code_dim)
    batch_size = 4
    dummy_features = torch.randn(batch_size, config.code_dim)

    # --- UPDATED TEST LOGIC ---
    # Unpack the four return values
    representation, codebook_loss, commitment_loss, code_entropy = codebook(dummy_features)

    # Assert the shape of the representation
    expected_repr_shape = (batch_size, config.code_dim)
    assert representation.shape == expected_repr_shape, \
        f"SharedCodebook expected representation shape {expected_repr_shape}, but got {representation.shape}"

    # Assert that the codebook loss is a scalar tensor
    assert isinstance(codebook_loss, torch.Tensor), "Codebook loss should be a tensor."
    assert codebook_loss.shape == (), f"Codebook loss should be a scalar, but got shape {codebook_loss.shape}"

    # Assert that the commitment loss is a scalar tensor
    assert isinstance(commitment_loss, torch.Tensor), "Commitment loss should be a tensor."
    assert commitment_loss.shape == (), f"Commitment loss should be a scalar, but got shape {commitment_loss.shape}"

    # Assert that the code entropy is a scalar tensor
    assert isinstance(code_entropy, torch.Tensor), "Code entropy should be a tensor."
    assert code_entropy.shape == (), f"Code entropy should be a scalar, but got shape {code_entropy.shape}"
    # --- END OF UPDATED LOGIC ---

    # Test that it fails with incorrect input dimensions
    with pytest.raises(ValueError):
        wrong_dim_features = torch.randn(batch_size, config.code_dim + 1)
        codebook(wrong_dim_features)

# --- Test the Assembled Agent ---

def test_perception_agent_initialization():
    """Tests that the PerceptionAgent can be initialized."""
    # This test is tricky because it modifies the config. Let's make it safer.
    # We create copies of the config to avoid side effects between tests.
    base_config = get_base_config()
    good_config = base_config.perception
    
    try:
        agent = PerceptionAgent(good_config)
        assert agent is not None, "Agent should not be None after initialization."
    except Exception as e:
        pytest.fail(f"PerceptionAgent initialization failed with error: {e}")

    # Test that initialization fails if dimensions don't match
    import copy
    faulty_config = copy.deepcopy(good_config)
    faulty_config.feature_dim = 128
    faulty_config.code_dim = 256 # Mismatch
    with pytest.raises(ValueError):
        PerceptionAgent(faulty_config)

def test_perception_agent_full_forward_pass():
    """Tests the full forward pass for correct return types and shapes."""
    config = get_base_config()
    agent = PerceptionAgent(config.perception)
    batch_size = 4
    image_size = config.env.image_size
    dummy_obs_batch = torch.randn(batch_size, 3, *image_size)
    
    # --- UPDATED TEST LOGIC ---
    # Unpack the four return values
    representation, codebook_loss, commitment_loss, code_entropy = agent(dummy_obs_batch)

    # Assert the shape of the representation
    expected_repr_shape = (batch_size, config.perception.code_dim)
    assert representation.shape == expected_repr_shape, \
        f"PerceptionAgent expected representation shape {expected_repr_shape}, but got {representation.shape}"

    # Assert that the loss terms are scalar tensors
    assert isinstance(codebook_loss, torch.Tensor), "Codebook loss should be a tensor."
    assert codebook_loss.shape == (), f"Codebook loss should be a scalar, but got shape {codebook_loss.shape}"

    assert isinstance(commitment_loss, torch.Tensor), "Commitment loss should be a tensor."
    assert commitment_loss.shape == (), f"Commitment loss should be a scalar, but got shape {commitment_loss.shape}"

    assert isinstance(code_entropy, torch.Tensor), "Code entropy should be a tensor."
    assert code_entropy.shape == (), f"Code entropy should be a scalar, but got shape {code_entropy.shape}"
    # --- END OF UPDATED LOGIC ---