import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from sample_factory.algo.utils.shared_buffers import (
    BufferMgr, alloc_trajectory_tensors, policy_device
)
from sample_factory.algo.utils.tensor_dict import TensorDict, clone_tensordict
from sample_factory.utils.utils import debug_log_every_n, log



class BatcherRamDataset(Dataset):
    """In-RAM dataset of tensor trajectories"""
    def __init__(self, max_size, rollout_length, env_info, rnn_spaces, device, 
                 share=False, seed=None):

        self._max_size = max_size
        self._rollout_length = rollout_length
        self._env_info = env_info
        self._rnn_spaces = rnn_spaces
        self._device = device
        self._share = share
        
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        # TODO: maybe add some kind of type converstion thing to reduce memory load

        self._data: TensorDict = alloc_trajectory_tensors(
            env_info=self._env_info,
            num_traj=self._max_size,
            rollout=self._rollout_length,
            rnn_spaces=self._rnn_spaces,
            device=self._device,
            share=self._share,
        )

        self._idx = 0
        self._cur_len = 0

    def __len__(self):
        return self._cur_len 

    @property
    def num_samples(self):
        return self._cur_len * self._rollout_length
    
    def add(self, trajectory: TensorDict):
        """
        Input is a trajectory TensorDict, the tensors have shape 
        (batch_size, rollout_length, ...)
        """
        # Check the batch size being added
        add_size = trajectory['dones'].shape[0]
        start, stop = self._idx, self._idx + add_size

        if stop > self._max_size:
            # If the trajectory is too big, split it into two
            # Amount of data that can be copied before reaching the max size
            num_until_max = self._max_size - start

            # Copy the first part up to the max size
            self._data[start:self._max_size] = trajectory[:num_until_max]

            # Wrap around and copy the remaining part
            num_remaining = add_size - num_until_max
            assert num_remaining < self._max_size,  "Trajectory larger than max dataset size"
            self._data[0:num_remaining] = trajectory[num_until_max:]
        else:
            # Copy the whole trajectory
            self._data[start:stop] = trajectory

        # Update indices
        self._idx = (self._idx + add_size) % self._max_size
        self._cur_len = min(self._cur_len + add_size, self._max_size)
    
        return True

    def __getitem__(self, idx):
        # NOTE TODO: for now using the full tajectory. Later should implement
        # some kind of timestep sampling and stitching (have to take care of
        # indexing)
        # Also maybe TODO:
        #  - after this: see how to efficiently move things aroudn on memory
        #  - add the reference to NM's repo for how we might want to implement this
        data_traj = clone_tensordict(self._data[idx])

        # NOTE: this is a custom if-then written just for NM's dreamer implementation
        if "is_first" in data_traj['obs']:
            data_traj['obs']['is_first'][0] = True
        
        return data_traj


class DatasetSampler(Sampler):
    """Custom trajectory sampler; inherits from torch.utils.data.Sampler"""
    def __init__(self, dataset):
        self._dataset = dataset

    def __iter__(self):
        # Implement your custom sampling logic here
        # For example, randomly sample indices
        import pdb; pdb.set_trace()
        indices = torch.randperm(len(self._dataset)).tolist()
        import pdb; pdb.set_trace()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)


