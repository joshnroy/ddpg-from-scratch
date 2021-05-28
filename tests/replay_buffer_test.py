import py
import pytest
import torch

from ddpg_from_scratch.replay_buffer import ReplayBuffer

def test_init_defaults():
    replay_buffer = ReplayBuffer(4)
    assert replay_buffer.batch_size == 512
    assert replay_buffer.state_dim == 4

@pytest.mark.parametrize("batch_size", [512, 128, 1, 2, 3, 17, 9000])
def test_init_batch_size(batch_size):
    replay_buffer = ReplayBuffer(4, batch_size=batch_size)
    assert replay_buffer.batch_size == batch_size

@pytest.mark.parametrize("batch_size", [0, -1, -200])
def test_init_wrong_batch_size(batch_size):
    with pytest.raises(ValueError):
        replay_buffer = ReplayBuffer(4, batch_size=batch_size)

@pytest.mark.parametrize("state_dim", [512, 128, 1, 2, 3, 17, 9000])
def test_init_state_dim(state_dim):
    replay_buffer = ReplayBuffer(state_dim)
    assert replay_buffer.state_dim == state_dim

@pytest.mark.parametrize("state_dim", [0, -1, -200])
def test_init_wrong_state_dim(state_dim):
    with pytest.raises(ValueError):
        replay_buffer = ReplayBuffer(state_dim)

@pytest.mark.parametrize("max_len", [0, -1, -200])
def test_init_wrong_max_len(max_len):
    with pytest.raises(ValueError):
        replay_buffer = ReplayBuffer(4, max_len=max_len)

@pytest.mark.parametrize("state_dim", [1, 3, 7, 10, 922])
def test_insert(state_dim):
    replay_buffer = ReplayBuffer(state_dim)

    assert len(replay_buffer) == 0
    for num_inserts in range(1, 11_000+1):
        replay_buffer.insert(torch.zeros(state_dim), torch.zeros(1), torch.zeros(1), torch.zeros(state_dim), torch.zeros(1))
        assert len(replay_buffer) == min(num_inserts, replay_buffer.max_len)

@pytest.mark.parametrize("batch_size", [1, 512, 200, 10, 45, 87])
def test_get_batches(batch_size):
    replay_buffer = ReplayBuffer(1, batch_size=batch_size)

    assert len(replay_buffer) == 0
    for num_inserts in range(0, replay_buffer.max_len):
        replay_buffer.insert(torch.zeros(1) + num_inserts, torch.zeros(1) + num_inserts, torch.zeros(1) + num_inserts, torch.zeros(1) + num_inserts, torch.zeros(1) + num_inserts)
    
    for x in replay_buffer.get_batches():
        assert len(x) == 5 # state, action, reward, next_state, done
        states, actions, rewards, next_states, dones = x
        assert len(states) == batch_size
        assert len(actions) == batch_size
        assert len(rewards) == batch_size
        assert len(next_states) == batch_size
        assert len(dones) == batch_size

@pytest.mark.parametrize("state_dim,insert_state_dim", [(1, 2), (1, 3), (1, -1), (4, 10), (4, -10), (4, 17)])
def wrong_state_dim_test_insert(state_dim, insert_state_dim):
    replay_buffer = ReplayBuffer(4)

    with pytest.raises(ValueError):
        replay_buffer.insert(torch.zeros(insert_state_dim), torch.zeros(1), torch.zeros(1), torch.zeros(state_dim), torch.zeros(1))

    with pytest.raises(ValueError):
        replay_buffer.insert(torch.zeros(state_dim), torch.zeros(1), torch.zeros(1), torch.zeros(insert_state_dim), torch.zeros(1))

@pytest.mark.parametrize("action_dim", [2, 6, 3, -1])
def wrong_action_dim_test_insert(action_dim):
    replay_buffer = ReplayBuffer(4)

    with pytest.raises(ValueError):
        replay_buffer.insert(torch.zeros(4), torch.zeros(action_dim), torch.zeros(1), torch.zeros(4), torch.zeros(1))

@pytest.mark.parametrize("reward_dim", [2, 6, 3, -1])
def wrong_reward_dim_test_insert(reward_dim):
    replay_buffer = ReplayBuffer(4)

    with pytest.raises(ValueError):
        replay_buffer.insert(torch.zeros(4), torch.zeros(1), torch.zeros(reward_dim), torch.zeros(4), torch.zeros(1))

@pytest.mark.parametrize("done_dim", [2, 6, 3, -1])
def wrong_done_dim_test_insert(done_dim):
    replay_buffer = ReplayBuffer(4)

    with pytest.raises(ValueError):
        replay_buffer.insert(torch.zeros(4), torch.zeros(1), torch.zeros(1), torch.zeros(4), torch.zeros(done_dim))