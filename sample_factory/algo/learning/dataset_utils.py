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
    def __init__(self, max_size, rollout_length, env_info, rnn_spaces, 
                 sample_whole_trajectories, sample_length,
                 device, share=False, seed=None):

        self._max_size = max_size
        self._rollout_length = rollout_length
        self._env_info = env_info
        self._rnn_spaces = rnn_spaces

        self._sample_whole_trajectories = sample_whole_trajectories
        self._sample_length = sample_length
        if self._sample_whole_trajectories:
            log.debug(
                f"[BatcherRamDataset init] Sampling whole trajectories (rollout_length={rollout_length}), "
                f"ignoring sample_length={sample_length}."
            )

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

    @staticmethod
    def _proess_data_traj(data_traj: TensorDict):
        # The trajectory tensor from the alloc_trajectory_tensors function
        # (in algo.util.shared_buffers) have an extra step for some of the 
        # keys, here we remove the final extra step from those keys
        keys_with_extra_step = ["obs", "rnn_states", "values", "valids"]
        for k in keys_with_extra_step:
            data_traj[k] = data_traj[k][:-1]
        
        return data_traj
    
    def _alloc_flat_trajectory_tensors(self, trajectory_length):
        """Allocate a "flat" trajectory tensor to store sample output"""
        output_data_traj = alloc_trajectory_tensors(
            env_info=self._env_info,
            num_traj=1,
            rollout=trajectory_length,
            rnn_spaces=self._rnn_spaces,
            device=self._device,  # TODO change this?
            share=self._share,
        )  # NOTE: this has one more dimensions

        # the allocated tensors have dimension (1, trajectory_length, ...), we 
        # will squeeze out the first dimension to be (trajectory_length, ...)
        def rec_squeeze(d):
            for k in d:
                if isinstance(d[k], dict):
                    d[k] = rec_squeeze(d[k])
                else:
                    d[k] = d[k].squeeze(0)
            return d
        
        output_data_traj = rec_squeeze(output_data_traj)
        return self._proess_data_traj(output_data_traj)

    def __getitem__(self, idx):
        # Get data trajectory from buffer
        cur_data_traj = self._data[idx]
        cur_data_traj = self._proess_data_traj(clone_tensordict(cur_data_traj))

        # If we are sampling full trajectories, just returned the full trajectory 
        if self._sample_whole_trajectories:            
            if "is_first" in cur_data_traj['obs']:
                # this is only used for NM's dreamerv3 implementation
                # https://github.com/im-ant/dreamerv3-torch/blob/main/tools.py#L346
                cur_data_traj['obs']['is_first'][0] = True
            return cur_data_traj
        
        # Sample sub-trajectories and stitch them together
        traj_len_total = self._sample_length
        output_data_traj = self._alloc_flat_trajectory_tensors(traj_len_total)

        # fill the output_data_traj with trajectories
        traj_len_sofar = 0
        while traj_len_sofar < traj_len_total:
            cur_traj_len = cur_data_traj['dones'].shape[0] 

            # start index and possible trajectory length to copy
            t_idx = self._rng.randint(0, cur_traj_len-1) \
                if (traj_len_sofar == 0) else 0
            cp_len = min(traj_len_total-traj_len_sofar, cur_traj_len - t_idx)

            # copy data
            output_data_traj[traj_len_sofar : traj_len_sofar + cp_len] = \
                cur_data_traj[t_idx : t_idx + cp_len]
                        
            if "is_first" in output_data_traj['obs']:
                # this is only used for NM's dreamerv3 implementation
                # https://github.com/im-ant/dreamerv3-torch/blob/main/tools.py#L346
                output_data_traj['obs']['is_first'][traj_len_sofar] = True

            # prep for next trajectory
            traj_len_sofar += cp_len
            if traj_len_sofar < traj_len_total:
                new_idx = self._rng.randint(0, len(self))
                cur_data_traj = self._proess_data_traj(
                    clone_tensordict(self._data[new_idx]))

        return output_data_traj

