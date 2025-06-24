import torch
import pytest
import numpy as np
from configs.base_config import get_base_config
from utils.replay_buffer import ReplayBuffer

def generate_dummy_transition(image_size=(64, 64)):
    obs = torch.rand(3, *image_size)
    action = np.random.randint(0, 17)
    reward = np.random.rand()
    next_obs = torch.rand(3, *image_size)
    terminated = False
    truncated = False
    return obs, action, reward, next_obs, terminated, truncated

def test_replay_buffer_initialization():
    config = get_base_config()
    buffer = ReplayBuffer(config.replay_buffer)
    assert buffer is not None
    assert len(buffer) == 0

def test_add_and_commit_episode():
    config = get_base_config()
    buffer = ReplayBuffer(config.replay_buffer)
    episode_length = 10

    for i in range(episode_length):
        is_terminal = (i == episode_length - 1)
        # CORRECTED: Unpack first, then call add.
        obs, action, reward, next_obs, _, _ = generate_dummy_transition()
        buffer.add(obs, action, reward, next_obs, terminated=is_terminal, truncated=False)
    
    assert len(buffer) == 1
    assert len(buffer.current_episode['actions']) == 0
    stored_episode = buffer.buffer[0]
    assert stored_episode['obs'].shape[0] == episode_length

def test_sampling_logic():
    config = get_base_config()
    config.replay_buffer.sequence_length = 10
    buffer = ReplayBuffer(config.replay_buffer)

    num_episodes = 5
    episode_len = 50
    for _ in range(num_episodes):
        for i in range(episode_len):
            is_terminal = (i == episode_len - 1)
            # CORRECTED: Unpack first, then call add.
            obs, action, reward, next_obs, _, _ = generate_dummy_transition()
            buffer.add(obs, action, reward, next_obs, terminated=is_terminal, truncated=False)

    assert len(buffer) == num_episodes

    sample = buffer.sample(batch_size=4, device='cpu')
    assert isinstance(sample, dict), "Sample should be a dictionary."
    
    seq_len = config.replay_buffer.sequence_length
    assert sample['obs'].shape[1] == seq_len

def test_edge_cases():
    config = get_base_config()
    config.replay_buffer.sequence_length = 50
    buffer = ReplayBuffer(config.replay_buffer)
    
    assert buffer.sample(batch_size=4, device='cpu') is None

    # Add a short episode
    episode_len_short = 10
    for i in range(episode_len_short):
        is_terminal = (i == episode_len_short - 1)
        # CORRECTED: Unpack first, then call add.
        obs, action, reward, next_obs, _, _ = generate_dummy_transition()
        buffer.add(obs, action, reward, next_obs, terminated=is_terminal, truncated=False)
    
    assert len(buffer) == 1
    assert buffer.sample(batch_size=4, device='cpu') is None
    
    # Add a long episode
    episode_len_long = 60
    for i in range(episode_len_long):
        is_terminal = (i == episode_len_long - 1)
        # CORRECTED: Unpack first, then call add.
        obs, action, reward, next_obs, _, _ = generate_dummy_transition()
        buffer.add(obs, action, reward, next_obs, terminated=is_terminal, truncated=False)

    assert len(buffer) == 2
    sample = buffer.sample(batch_size=4, device='cpu')
    assert sample is not None