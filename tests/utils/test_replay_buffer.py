import torch
import pytest
import numpy as np
from configs.base_config import get_base_config
from utils.replay_buffer import ReplayBuffer

def generate_dummy_transition(image_size=(64, 64), state_dim=256):
    obs = torch.rand(3, *image_size)
    action = np.random.randint(0, 17)
    log_prob = np.random.randn()
    reward = np.random.rand()
    next_obs = torch.rand(3, *image_size)
    terminated = False
    truncated = False
    state_repr = torch.randn(state_dim)
    return obs, action, log_prob, reward, next_obs, terminated, truncated, state_repr

def test_replay_buffer_initialization():
    config = get_base_config()
    buffer = ReplayBuffer(config.replay_buffer)
    assert buffer is not None
    assert len(buffer) == 0

def test_add_and_commit_episode():
    config = get_base_config()
    buffer = ReplayBuffer(config.replay_buffer)
    episode_length = 10
    state_dim = config.perception.code_dim

    for i in range(episode_length):
        is_terminal = (i == episode_length - 1)
        # Unpack all values and pass them to add
        obs, action, log_prob, reward, next_obs, _, _, state_repr = generate_dummy_transition(state_dim=state_dim)
        buffer.add(obs, action, log_prob, reward, next_obs, is_terminal, False, state_repr)
    
    assert len(buffer) == 1
    assert len(buffer.current_episode['actions']) == 0
    stored_episode = buffer.buffer[0]
    assert stored_episode['obs'].shape[0] == episode_length

def test_sampling_logic():
    config = get_base_config()
    config.replay_buffer.sequence_length = 10
    buffer = ReplayBuffer(config.replay_buffer)
    state_dim = config.perception.code_dim

    num_episodes = 5
    episode_len = 50
    for _ in range(num_episodes):
        for i in range(episode_len):
            is_terminal = (i == episode_len - 1)
            # Unpack all values and pass them to add
            obs, action, log_prob, reward, next_obs, _, _, state_repr = generate_dummy_transition(state_dim=state_dim)
            buffer.add(obs, action, log_prob, reward, next_obs, is_terminal, False, state_repr)

    assert len(buffer) == num_episodes

    sample = buffer.sample(batch_size=4, device='cpu')
    assert isinstance(sample, dict), "Sample should be a dictionary."
    
    seq_len = config.replay_buffer.sequence_length
    assert sample['obs'].shape[1] == seq_len

def test_edge_cases():
    config = get_base_config()
    config.replay_buffer.sequence_length = 50
    buffer = ReplayBuffer(config.replay_buffer)
    state_dim = config.perception.code_dim
    
    assert buffer.sample(batch_size=4, device='cpu') is None

    # Add a short episode
    episode_len_short = 10
    for i in range(episode_len_short):
        is_terminal = (i == episode_len_short - 1)
        # Unpack all values and pass them to add
        obs, action, log_prob, reward, next_obs, _, _, state_repr = generate_dummy_transition(state_dim=state_dim)
        buffer.add(obs, action, log_prob, reward, next_obs, is_terminal, False, state_repr)
    
    assert len(buffer) == 1
    assert buffer.sample(batch_size=4, device='cpu') is None
    
    # Add a long episode
    episode_len_long = 60
    for i in range(episode_len_long):
        is_terminal = (i == episode_len_long - 1)
        # Unpack all values and pass them to add
        obs, action, log_prob, reward, next_obs, _, _, state_repr = generate_dummy_transition(state_dim=state_dim)
        buffer.add(obs, action, log_prob, reward, next_obs, is_terminal, False, state_repr)

    assert len(buffer) == 2
    sample = buffer.sample(batch_size=4, device='cpu')
    assert sample is not None

def test_clear_buffer():
    """Tests that the clear method empties both internal buffers."""
    config = get_base_config()
    config.replay_buffer.sequence_length = 10
    buffer = ReplayBuffer(config.replay_buffer)
    state_dim = config.perception.code_dim

    # Add a long episode that should populate both internal deques
    episode_len = 50
    for i in range(episode_len):
        is_terminal = (i == episode_len - 1)
        obs, action, log_prob, reward, next_obs, _, _, state_repr = generate_dummy_transition(state_dim=state_dim)
        buffer.add(obs, action, log_prob, reward, next_obs, is_terminal, False, state_repr)
    
    assert len(buffer) == 1
    assert len(buffer.valid_buffer) == 1
    
    buffer.clear()
    
    assert len(buffer) == 0
    assert len(buffer.valid_buffer) == 0
    assert len(buffer.current_episode['actions']) == 0

def test_eviction_synchronization():
    """
    Tests that when an episode is evicted from the main buffer, it is also
    correctly removed from the valid_buffer to prevent sampling stale data.
    This specifically tests the case where a short episode is evicted, which
    was the source of a bug.
    """
    config = get_base_config()
    config.replay_buffer.capacity = 2
    config.replay_buffer.sequence_length = 20
    buffer = ReplayBuffer(config.replay_buffer)
    state_dim = config.perception.code_dim

    # Helper to add an episode of a specific length
    def add_episode(length, is_terminal=True):
        for i in range(length):
            terminal = is_terminal and (i == length - 1)
            transition = generate_dummy_transition(state_dim=state_dim)
            buffer.add(*transition[:5], terminal, False, transition[7])
        return buffer.buffer[-1] # Return the committed episode dict

    # 1. Add a short episode. It should NOT be in valid_buffer.
    short_episode = add_episode(length=10)
    assert len(buffer.valid_buffer) == 0

    # 2. Add a long episode. It SHOULD be in valid_buffer.
    long_episode_1 = add_episode(length=30)
    assert len(buffer) == 2
    assert len(buffer.valid_buffer) == 1

    # 3. Add another long episode. This evicts the short_episode.
    # The key test: `valid_buffer` must not contain stale references.
    long_episode_2 = add_episode(length=30)
    assert len(buffer) == 2 # Capacity is 2
    assert len(buffer.valid_buffer) == 2
    # We must check for the *identity* of the episode object, not equality.
    # Using `in` (which uses `==`) on dicts of numpy arrays is ambiguous
    # and raises a ValueError if shapes mismatch or are non-scalar.
    assert not any(e is short_episode for e in buffer.buffer)
    assert any(e is long_episode_1 for e in buffer.valid_buffer)
    assert any(e is long_episode_2 for e in buffer.valid_buffer)