import torch
from random import shuffle

class ReplayBuffer():
    """ Holds all of the states, actions, rewards, next states, and dones that have been stored. Also responsible for shuffling and getting batches during training time
    """
    def __init__(self, state_dim: int, batch_size: int = 512, max_len: int = 10_000):
        """Creates the replay buffer

        Args:
            state_dim (int): The dimensionality (currently only supports 1d) of the state
            batch_size (int, optional): Size that the batches should return. Defaults to 512.
            max_len (int, optional): The maximum length of the buffer

        Raises:
            ValueError: If the batch size is less than 1
            ValueError: If the state dim is less than 1
            ValueError: If the max len is less than 1
        """
        if batch_size < 1:
            raise ValueError("Batch size must be above 1")
        if state_dim < 1:
            raise ValueError("State dim must be at least 1")
        if max_len < 1:
            raise ValueError("State dim must be at least 1")
        
        self._state_dim = state_dim
        self._batch_size = batch_size
        self._max_len = max_len

        # state, action, reward, next_state, done
        self._buffer = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}

    def insert(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor):
        """Inserts a tuple into the array buffer

        Args:
            state (torch.Tensor): The state tensor
            action (torch.Tensor): The action tensor
            reward (torch.Tensor): The reward tensor
            next_state (torch.Tensor): The next state tensor
            done (torch.Tensor): The done tensor

        Raises:
            ValueError: If the state dim is wrong
            ValueError: If the next state dim is wrong
            ValueError: If the action dim is not 1
            ValueError: If the reward dim is not 1
            ValueError: If the done dim is not done
        """
        if len(state.shape) != 1 and state.shape[0] != self.state_dim:
            raise ValueError("State is not dimension", self.state_dim)
        if len(next_state.shape) != 1 and next_state.shape[0] != self.state_dim:
            raise ValueError("Next State is not dimension", self.state_dim)
        if len(action.shape) != 1 and action.shape[0] != 1:
            raise ValueError("Action is not dimension", 1)
        if len(reward.shape) != 1 and reward.shape[0] != 1:
            raise ValueError("Reward is not dimension", 1)
        if len(done.shape) != 1 and done.shape[0] != 1:
            raise ValueError("Done is not dimension", 1)

        if len(self) >= self.max_len:
            for k in self._buffer:
                self._buffer[k] = self._buffer[k][-(self.max_len-1):]

        self._buffer['state'].append(state)
        self._buffer['action'].append(action)
        self._buffer['reward'].append(reward)
        self._buffer['next_state'].append(next_state)
        self._buffer['done'].append(done)

    def get_batches(self):
        for idx in range(0, len(self)-self.batch_size, self.batch_size):
            yield (self._buffer['state'][idx:idx+self.batch_size], self._buffer['action'][idx:idx+self.batch_size], self._buffer['reward'][idx:idx+self.batch_size], self._buffer['next_state'][idx:idx+self.batch_size], self._buffer['done'][idx:idx+self.batch_size])
        idxs = [i for i in range(len(self))]
        shuffle(idxs)

        # Shuffle at the end of each epoch
        self._buffer['state'] = [self._buffer['state'][idx] for idx in idxs]
        self._buffer['action'] = [self._buffer['action'][idx] for idx in idxs]
        self._buffer['reward'] = [self._buffer['reward'][idx] for idx in idxs] 
        self._buffer['next_state'] = [self._buffer['next_state'][idx] for idx in idxs]
        self._buffer['done'] = [self._buffer['done'][idx] for idx in idxs]

    @property
    def batch_size(self):
        """Returns the batch size

        Returns:
            int: the batch size
        """
        return self._batch_size

    @property
    def state_dim(self):
        """Returns the state dimension

        Returns:
            int: the state dim
        """
        return self._state_dim

    def __len__(self):
        return len(self._buffer['state'])

    @property
    def max_len(self):
        return self._max_len