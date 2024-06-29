"""
This is a simplified version of the stable-baselines3 replay buffer taken from
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py

I've removed unneeded functionality and put all dependencies into a single file.
"""
import copy
import random
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Tuple, Union

import numpy as np
import psutil
import torch
from gymnasium import spaces
from per_tree import SumTree
from torch.utils.data import DataLoader, TensorDataset
import time
import timeit


class ReplayBufferSamples(NamedTuple):
    """Container for replay buffer samples."""

    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


def get_obs_shape(observation_space: spaces.Space) -> Union[tuple[int, ...], Dict[str, tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).
    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        if type(observation_space.n) in [tuple, list, np.ndarray]:
            return tuple(observation_space.n)
        else:
            return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]
    else:
        # print(type(observation_space))
        # print(type(spaces.Box(low=0., high=1., shape=(2,), dtype=np.float32)))
        # print(isinstance(observation_space, spaces.Box))
        # print(isinstance(observation_space, spaces.box.Box))
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.
    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.
    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to torch.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)
        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default
        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def get_partition(self, start_idx: int, end_idx: int) -> ReplayBufferSamples:
        """
        Get a partition of the replay buffer.

        :param start_idx: Start index of the partition
        :param end_idx: End index of the partition
        :return: A ReplayBufferSamples object containing the partitioned transitions
        """
        # Handle wrap-around if end_idx exceeds buffer size
        if start_idx >= self.buffer_size or end_idx > self.buffer_size:
            raise IndexError("Indices exceed buffer size.")

        if start_idx >= end_idx:
            raise ValueError("Start index must be less than end index.")

        partition_indices = np.arange(start_idx, end_idx) % self.buffer_size

        return self._get_samples(partition_indices)

    def get_n_partitions(self, n: int) -> list:
        """
        Divide the replay buffer into n partitions.

        :param n: Number of partitions
        :return: List of ReplayBufferSamples objects containing the partitions
        """
        if n <= 0:
            raise ValueError("Number of partitions must be greater than 0.")

        partition_size = self.buffer_size // n
        partitions = []

        for i in range(n):
            start_idx = i * partition_size
            end_idx = (i + 1) * partition_size if i < n - 1 else self.buffer_size
            partitions.append(self.get_partition(start_idx, end_idx))

        return partitions

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        :param batch_size: Number of element to sample
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
    
    def quantization_scores(self, observations, actions, qf, logs=None, quantization_type="fp16", quantization_rand_prunning_fraction=.5, critical_value_eval_batch_size=256) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        dataset = TensorDataset(observations)
        dataloader = DataLoader(dataset, batch_size=critical_value_eval_batch_size, shuffle=False)

        # Initialize lists to store results
        fp32_qf_a_values_list = []
        quantized_qf_a_values_list = []

        # Process batches
        for obs_batch in dataloader:
            obs_batch = obs_batch[0]
            assert isinstance(obs_batch, torch.Tensor), "obs_batch is not a tensor"
            # Get the critical transitions
            with torch.no_grad():
                # Get original model size values
                fp32_qf_a_values_batch = qf(obs_batch).view(-1, obs_batch.size(0))
                fp32_qf_a_values_list.append(fp32_qf_a_values_batch)
                if quantization_type == "int8":
                    # if qf_quantized is None:
                    qf.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                    torch.backends.quantized.engine = "fbgemm"
                    qf_prepared = torch.quantization.prepare(copy.deepcopy(qf), inplace=False)
                    qf_prepared = qf_prepared.to('cpu')
                    # Use 10 Samples NOTE: This could throw error so do fraction of obs tensor flat later
                    calibration_data_obs = obs_batch[:10]
                    qf_prepared(calibration_data_obs.to('cpu'))
                    qf_quantized = torch.quantization.convert(qf_prepared, inplace=False)
                    activation_observer = qf_prepared.fc1.activation_post_process
                    scale_obs, zero_point_obs = activation_observer.calculate_qparams()
                    inference_obs_data_q = torch.quantize_per_tensor(obs_batch.to('cpu'), scale=scale_obs, zero_point=zero_point_obs, dtype=torch.quint8).to("cpu")
                    quantized_qf_a_values_batch = qf_quantized(inference_obs_data_q).view(-1, inference_obs_data_q.size(0)).dequantize()
                elif quantization_type == "fp16":
                    fp16_qf = copy.deepcopy(qf).half()
                    quantized_qf_a_values_batch = fp16_qf(obs_batch.half()).view(-1, obs_batch.size(0))
                elif quantization_type == "rand_prunning":
                    # This will randomly reset model weights to 0 after every logging batch
                    assert quantization_rand_prunning_fraction < 1 and quantization_rand_prunning_fraction >= 0, "Fraction of weights to be pruned must be greater than 0 and less than 1."
                    def prune_weights(state_dict, indices, all_weights):
                        # Set the selected indices to zero
                        all_weights_flat = all_weights.clone()
                        all_weights_flat[indices] = 0
                        # Reconstruct the state_dict with pruned weights
                        offset = 0
                        for key in state_dict.keys():
                            numel = state_dict[key].numel()
                            state_dict[key] = all_weights_flat[offset:offset + numel].view_as(state_dict[key])
                            offset += numel
                        return state_dict

                    pruned_qf = copy.deepcopy(qf)
                    state_dict_1 = pruned_qf.state_dict()
                    # Concatenate all the weights into a single tensor
                    # Just use one of the networks to remove the same weights in both networks later
                    all_weights_1 = torch.cat([w.flatten() for w in state_dict_1.values()])
                    n = all_weights_1.numel()
                    m = int(round(n * quantization_rand_prunning_fraction))
                    if m > 0:
                        # Randomly select indices to prune
                        indices = np.random.choice(n, m, replace=False)
                        state_dict_1 = prune_weights(state_dict_1, indices, all_weights_1)
                        # Load the pruned weights back into the models
                        pruned_qf.load_state_dict(state_dict_1)
                    quantized_qf_a_values_batch = pruned_qf(obs_batch).view(-1, obs_batch.size(0))
                else:
                    NotImplementedError(f"Quantization Type not defined for {quantization_type}")
                quantized_qf_a_values_list.append(quantized_qf_a_values_batch)

        def pad_with_nan(tensor, target_size):
            # Create a new tensor filled with NaN values
            padded_tensor = torch.full((tensor.size(0), target_size), float('nan'))
            # Copy the original tensor's values into the new tensor
            padded_tensor[:, :tensor.size(1)] = tensor
            return padded_tensor

        def remove_nan_columns(tensor):
            # Create a mask for non-NaN values
            non_nan_mask = ~torch.isnan(tensor)
            # Remove all NaN values while maintaining the structure of the tensor
            non_nan_values = tensor[non_nan_mask]
            # Calculate the number of valid (non-NaN) columns
            valid_cols = non_nan_mask.sum(dim=1).max().item()
            # Reshape the non-NaN values back to the original number of rows with the calculated valid columns
            cleaned_tensor = non_nan_values.view(tensor.size(0), valid_cols)
            return cleaned_tensor
        
        # Pad the tensors to match the maximum size for tensor concatenation
        fp32_qf_a_values_list = [pad_with_nan(t, critical_value_eval_batch_size) for t in fp32_qf_a_values_list]
        quantized_qf_a_values_list = [pad_with_nan(t, critical_value_eval_batch_size) for t in quantized_qf_a_values_list]

        # Concatenate batch results and reshape them to remove Nan values in next step by flattening all dims apart from first dim
        # From https://stackoverflow.com/questions/64594493/filter-out-nan-values-from-a-pytorch-n-dimensional-tensor
        # Concatenate along the columns
        fp32_qf_a_values = torch.cat(fp32_qf_a_values_list, dim=1)
        quantized_qf_a_values = torch.cat(quantized_qf_a_values_list, dim=1)

        fp32_qf_a_values = remove_nan_columns(fp32_qf_a_values)
        quantized_qf_a_values = remove_nan_columns(quantized_qf_a_values)

        # Reshape the tensor
        actions = actions.view(-1, 1)

        # Transpose
        fp32_qf_a_values = fp32_qf_a_values.transpose(0, 1)
        quantized_qf_a_values = quantized_qf_a_values.transpose(0, 1)
        # Gather Q-values for the specific actions
        fp32_q_values = fp32_qf_a_values.gather(1, actions.to("cpu")).squeeze(1)
        quantized_q_values = quantized_qf_a_values.gather(1, actions.to("cpu")).squeeze(1)

        # Compute critical values
        assert fp32_q_values.size() == quantized_q_values.size(), "Tensors must have the same size"

        # Compute element-wise differences
        differences = fp32_q_values - quantized_q_values
        # Compute element-wise squared differences
        squared_differences = differences.pow(2)
        # Compute the RR scores for each sample
        critical_scores = torch.sqrt(squared_differences)

        if logs is not None:
            logs["score_logs/max_critical_scores"] = max(critical_scores).item()
            logs["score_logs/min_critical_scores"] = min(critical_scores).item()
            logs["score_logs/mean_critical_scores"] = torch.mean(critical_scores).item()
            logs["score_logs/std_critical_scores"] = torch.std(critical_scores).item()
            logs["score_logs/median_critical_scores"] = torch.median(critical_scores).item()

            return critical_scores, {}

        return critical_scores, logs


    def quantization_transitions(self, qf, quantization_type="fp16", quantization_rand_prunning_fraction=.5, epsilon=.1, critical_value_eval_batch_size=256) -> ReplayBufferSamples:
        """
        Get the transitions that are critical for the agent to learn in the previous environment by the critcal quantization value (RRscore).
        """
        logs = {}
        # TODO: Only works for one environement at a time
        # Get all the transitions from the replay buffer
        transitions = self.sample(self.pos)
        observations = transitions.observations
        actions = transitions.actions
        next_observations = transitions.next_observations
        dones = transitions.dones
        rewards = transitions.rewards
        
        critical_scores, logs = self.quantization_scores(observations, actions, qf, logs, quantization_type, quantization_rand_prunning_fraction, critical_value_eval_batch_size)
        assert logs != {}, "logs is empty."

        mean_critical_score = torch.mean(critical_scores).item()
        mask = critical_scores > mean_critical_score
        indices = torch.nonzero(mask).squeeze()
        # Get the elements that are larger than the threshold
        # masked_critical_scores = critical_scores[mask]

        # Ensure k does not exceed the number of elements
        # k = min(k, len(sorted_indices))
        # top_k_indices = sorted_indices[:k]

        # Filter the transitions
        data = (
            observations[indices],
            actions[indices],
            next_observations[indices],
            dones[indices],
            rewards[indices],
        )
        return ReplayBufferSamples(*tuple(data)), logs
    

    def fatal_scores(self, observations, qf, logs=None, critical_value_eval_batch_size=256) -> Tuple[torch.Tensor, Dict[str, Any]]:
        dataset = TensorDataset(observations)
        dataloader = DataLoader(dataset, batch_size=critical_value_eval_batch_size, shuffle=False)

        # Initialize lists to store results
        q_values = []

        # Process batches
        for obs_batch in dataloader:
            obs_batch = obs_batch[0]
            assert isinstance(obs_batch, torch.Tensor), "obs_batch is not a tensor"
            # Get the critical transitions
            with torch.no_grad():
                # Get original model size values
                q_values_batch = qf(obs_batch).view(-1, obs_batch.size(0))
                q_values.append(q_values_batch)

        def pad_with_nan(tensor, target_size):
            # Create a new tensor filled with NaN values
            padded_tensor = torch.full((tensor.size(0), target_size), float('nan'))
            # Copy the original tensor's values into the new tensor
            padded_tensor[:, :tensor.size(1)] = tensor
            return padded_tensor

        def remove_nan_columns(tensor):
            # Create a mask for non-NaN values
            non_nan_mask = ~torch.isnan(tensor)
            # Remove all NaN values while maintaining the structure of the tensor
            non_nan_values = tensor[non_nan_mask]
            # Calculate the number of valid (non-NaN) columns
            valid_cols = non_nan_mask.sum(dim=1).max().item()
            # Reshape the non-NaN values back to the original number of rows with the calculated valid columns
            cleaned_tensor = non_nan_values.view(tensor.size(0), valid_cols)
            return cleaned_tensor
        
        # Pad the tensors to match the maximum size for tensor concatenation
        q_values = [pad_with_nan(t, critical_value_eval_batch_size) for t in q_values]
        # Concatenate batch results and reshape them to remove Nan values in next step by flattening all dims apart from first dim
        # From https://stackoverflow.com/questions/64594493/filter-out-nan-values-from-a-pytorch-n-dimensional-tensor
        # Concatenate along the columns
        q_values = torch.cat(q_values, dim=1)
        q_values = remove_nan_columns(q_values)

        # Compute critical scores elementwise
        # TODO: Possibly artificially generate transitions with the highest and lowest critical scores and place them in the global buffer
        critical_scores = torch.abs(torch.max(q_values, dim=0)[0] - torch.min(q_values, dim=0)[0]).unsqueeze(1).squeeze()  # Shape: [num_obs]

        if logs is not None:
            logs["score_logs/max_critical_scores"] = max(critical_scores).item()
            logs["score_logs/min_critical_scores"] = min(critical_scores).item()
            logs["score_logs/mean_critical_scores"] = torch.mean(critical_scores).item()
            logs["score_logs/std_critical_scores"] = torch.std(critical_scores).item()
            logs["score_logs/median_critical_scores"] = torch.median(critical_scores).item()

            return critical_scores, {}

        return critical_scores, logs
    
    def fatal_transitions(self, qf, k=1000, critical_value_eval_batch_size=256) -> ReplayBufferSamples:
        """
        Get the transitions that are critical for the agent to learn in the previous environment by the critcal quantization value (RRscore).

        Parameters:
            qf (QNetwork object): The Q-function used to calculate the critical quantization value.
            k (int): The number of transitions to select as critical. Default is 1000.
            critical_value_eval_batch_size (int): The batch size used to avoid OOM when passing obs through qf. Default is 256.

        Returns:
            ReplayBufferSamples: A tuple containing filtered transitions.
            logs: A dictionary of logs.
        """
        logs = {}
        # TODO: Only works for one environement at a time
        # Get all the transitions from the replay buffer
        transitions = self.sample(self.pos)

        observations = transitions.observations
        actions = transitions.actions
        next_observations = transitions.next_observations
        dones = transitions.dones
        rewards = transitions.rewards

        
        critical_scores, logs = self.fatal_scores(observations, qf, logs, k, critical_value_eval_batch_size)
        assert logs != {}, "logs is empty."

        mean_critical_score = torch.mean(critical_scores).item()
        mask = critical_scores > mean_critical_score
        indices = torch.nonzero(mask).squeeze()
        # Get the elements that are larger than the threshold
        # masked_critical_scores = critical_scores[mask]

        # Ensure k does not exceed the number of elements
        # k = min(k, len(sorted_indices))
        # top_k_indices = sorted_indices[:k]

        # Filter the transitions
        data = (
            observations[indices],
            actions[indices],
            next_observations[indices],
            dones[indices],
            rewards[indices],
        )
        return ReplayBufferSamples(*tuple(data)), logs
    
    def coverage_transitions(self, k=40000, d=.5) -> ReplayBufferSamples:
        """
        Model-agnostic sampler
        """
        logs = {}
        # TODO: Only works for one environement at a time
        # Get all the transitions from the replay buffer
        transitions = self.sample(self.pos)
        observations = transitions.observations
        actions = transitions.actions
        next_observations = transitions.next_observations
        dones = transitions.dones
        rewards = transitions.rewards
                                                                                                                                                                                                                                                                                                                                         
        # Convert transitions into a list of lists
        transition_list = []
        for i in range(observations.shape[0]):
            transition = np.concatenate([
                observations[i].cpu().numpy().flatten(),
                actions[i].cpu().numpy().flatten(),
                next_observations[i].cpu().numpy().flatten(),
                dones[i].cpu().numpy().flatten(),
                rewards[i].cpu().numpy().flatten()
            ])
            transition_list.append(transition)

        transition_list = np.array(transition_list)
        s_coverage = np.zeros(len(transition_list))

        # Compute the distance between transitions
        def compute_distance(e1, e2):
            # print(f"==>> np.linalg.norm(e1 - e2): {np.linalg.norm(e1 - e2)}")
            return np.linalg.norm(e1 - e2)

        start_time = time.time()

        print(f"==>> transitions.shape: {transition_list.shape}")
        for i, e_i in enumerate(transition_list):
            count = 0
            for j, e_j in enumerate(transition_list):
                if i != j and compute_distance(e_i, e_j) < d:
                    count += 1
            s_coverage[i] = -count

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Execution time: {execution_time} seconds")
        print(f"==>> s_coverage: {s_coverage[:100]}")

        # Compute the pairwise distances between all transitions
        start_time = time.time()
        pairwise_distances = np.linalg.norm(transition_list[:, np.newaxis] - transition_list[np.newaxis, :], axis=-1)

        print(f"==>> pairwise_distances: {pairwise_distances[:100]}")
        print(f"==>> transitions.shape: {transition_list.shape}")
        print(f"==>> pairwise_distances.shape: {pairwise_distances.shape}")

        # Count the number of transitions within the distance threshold for each transition
        within_distance = (pairwise_distances < d) & (pairwise_distances > 0)  # Exclude self-distance
        print(f"==>> within_distance: {within_distance[:100]}")
        print(f"==>> within_distance.shape: {within_distance.shape}")

        # Sum the number of such transitions for each transition
        s_coverage = -np.sum(within_distance, axis=1)

        end_time = time.time()
        print(f"Time taken to compute pairwise distances: {end_time - start_time} seconds")
        print(f"==>> s_coverage: {s_coverage}")

        sorted_scores, sorted_indices = torch.sort(s_coverage, descending=True)

        # Ensure k does not exceed the number of elements
        k = min(k, len(sorted_indices))
        indices = sorted_indices[:k]
        print(f"==>> indices.shape: {indices.shape}")
        

        exit()

        logs["score_logs/max_critical_scores"] = max(critical_scores).item()
        logs["score_logs/min_critical_scores"] = min(critical_scores).item()
        logs["score_logs/mean_critical_scores"] = torch.mean(critical_scores).item()
        logs["score_logs/std_critical_scores"] = torch.std(critical_scores).item()
        logs["score_logs/median_critical_scores"] = torch.median(critical_scores).item()

        # Get indices where critical_scores > epsilon
        indices = torch.nonzero(critical_scores > epsilon).flatten()

        # Filter the transitions
        data = (
            observations[indices],
            actions[indices],
            next_observations[indices],
            dones[indices],
            rewards[indices],
        )
        return ReplayBufferSamples(*tuple(data)), logs
    
    def bayesian_transitions(self, lambdas: dict, bayesian_update_strats: list[str], qf, k, d, quantization_type, quantization_rand_prunning_fraction) -> Tuple[ReplayBufferSamples, dict, dict]:
        """
        Get the transitions with highest linear score for different selection strategies
        """
        transitions = self.sample(self.pos)
        observations = transitions.observations
        actions = transitions.actions
        next_observations = transitions.next_observations
        dones = transitions.dones
        rewards = transitions.rewards
        logs = {}

        scores = torch.zeros(observations.shape[0], dtype=torch.float32)
        if "fatality" in bayesian_update_strats:
            fatal_scores, logs = self.fatal_scores(observations, qf, logs)
            fatal_scores = fatal_scores * lambdas["fatality"]
            scores += fatal_scores
        if "quantization" in bayesian_update_strats:
            quantization_scores, logs = self.quantization_scores(observations, actions, qf, logs, quantization_type, quantization_rand_prunning_fraction)
            quantization_scores = quantization_scores * lambdas["quantization"]
            scores += quantization_scores
        if "coverage" in bayesian_update_strats:
            NotImplemented

        # Use all transitions that have a score above the mean
        mean_score = torch.mean(scores).item()
        mask = scores >= mean_score
        indices = torch.nonzero(mask).squeeze()
        # Filter the transitions
        data = (
            observations[indices],
            actions[indices],
            next_observations[indices],
            dones[indices],
            rewards[indices],
        )
        return ReplayBufferSamples(*tuple(data)), logs
    
    def pairwiseRR(self, qf1, qf2, external_buffer, epsilon=.1, critical_value_eval_batch_size=256) -> ReplayBufferSamples:
        """
        Get the transitions that are critical for the agent to learn in the previous environment by the critcal quantization value (RRscore).
        """
        # TODO: Only works for one environement at a time
        # Loop through multiple paritions of the replay buffers to avoid OOM leaks
        # transitions = self.sample(self.size())
        # external_transitions = external_buffer.sample(external_buffer.size())
        print(f"==>> external_buffer.pos: {external_buffer.pos}")
        print(f"==>> self.pos: {self.pos}")
        size = max(self.pos, external_buffer.pos)
        partition_count = max(size // 100, 1)  # Adjust partition count as needed
        transition_partitions = self.get_n_partitions(partition_count)
        external_transition_partitions = external_buffer.get_n_partitions(partition_count)

        all_observations = []
        all_actions = []
        all_next_observations = []
        all_dones = []
        all_rewards = []

        for transitions, external_transitions in zip(transition_partitions, external_transition_partitions):
            # Combine Transitions
            observations = torch.cat((transitions.observations, external_transitions.observations), dim=0)
            actions = torch.cat((transitions.actions, external_transitions.actions), dim=0)
            next_observations = torch.cat((transitions.next_observations, external_transitions.next_observations), dim=0)
            dones = torch.cat((transitions.dones, external_transitions.dones), dim=0)
            rewards = torch.cat((transitions.rewards, external_transitions.rewards), dim=0)

            dataset = TensorDataset(observations.cpu()) # Move to CPU
            dataloader = DataLoader(dataset, batch_size=critical_value_eval_batch_size, shuffle=False)

            # Initialize lists to store results
            qf1_values_list = []
            qf2_values_list = []

            # Process batches
            for obs_batch in dataloader:
                obs_batch = obs_batch[0].to(qf1.device)  # Move back to GPU for inference
                assert isinstance(obs_batch, torch.Tensor), "obs_batch is not a tensor"
                # Get the critical transitions
                with torch.no_grad():
                    # Get original model size values
                    qf1_values_batch = qf1(obs_batch).view(-1, obs_batch.size(0))
                    qf2_values_batch = qf2(obs_batch).view(-1, obs_batch.size(0))
                    qf1_values_list.append(qf1_values_batch.cpu())  # Move back to CPU
                    qf2_values_list.append(qf2_values_batch.cpu())  # Move back to CPU

            def pad_with_nan(tensor, target_size):
                # Create a new tensor filled with NaN values
                padded_tensor = torch.full((tensor.size(0), target_size), float('nan'))
                # Copy the original tensor's values into the new tensor
                padded_tensor[:, :tensor.size(1)] = tensor
                return padded_tensor
            
            def remove_nan_columns(tensor):
                # Create a mask for non-NaN values
                non_nan_mask = ~torch.isnan(tensor)
                # Remove all NaN values while maintaining the structure of the tensor
                non_nan_values = tensor[non_nan_mask]
                # Calculate the number of valid (non-NaN) columns
                valid_cols = non_nan_mask.sum(dim=1).max().item()
                # Reshape the non-NaN values back to the original number of rows with the calculated valid columns
                cleaned_tensor = non_nan_values.view(tensor.size(0), valid_cols)
                return cleaned_tensor
            
            # Pad the tensors to match the maximum size for tensor concatenation
            qf1_values_list = [pad_with_nan(t, critical_value_eval_batch_size) for t in qf1_values_list]
            qf2_values_list = [pad_with_nan(t, critical_value_eval_batch_size) for t in qf2_values_list]

            # Concatenate batch results and reshape them to remove Nan values in next step by flattening all dims apart from first dim
            # From https://stackoverflow.com/questions/64594493/filter-out-nan-values-from-a-pytorch-n-dimensional-tensor
            # Concatenate along the columns
            qf1_values = torch.cat(qf1_values_list, dim=1)
            qf2_values = torch.cat(qf2_values_list, dim=1)

            # fp32_qf_a_values = fp32_qf_a_values[~fp32_qf_a_values.isnan()]
            # quantized_qf_a_values = quantized_qf_a_values[~quantized_qf_a_values.isnan()]
            qf1_values = remove_nan_columns(qf1_values)
            qf2_values = remove_nan_columns(qf2_values)

            # Reshape the tensor
            actions = actions.view(-1, 1)

            # Transpose and Gather Q-values for the specific actions
            qf1_values = qf1_values.transpose(0, 1).gather(1, actions).squeeze(1)
            qf2_values = qf2_values.transpose(0, 1).gather(1, actions).squeeze(1)

            # Compute critical values
            assert qf1_values.size() == qf1_values.size(), "Tensors must have the same size"

            # Compute element-wise differences
            differences = qf1_values - qf2_values
            # Compute element-wise squared differences
            squared_differences = differences.pow(2)
            # Compute the RR scores for each sample
            rr_scores = torch.sqrt(squared_differences)
            print(f"==>> max rr_scores: {max(rr_scores)}")
            print(f"==>> min rr_scores: {min(rr_scores)}")
            
            # Get indices where rr_scores > epsilon
            indices = torch.nonzero(rr_scores > epsilon).flatten()

            # Filter the transitions
            all_observations.append(observations[indices])
            all_actions.append(actions[indices])
            all_next_observations.append(next_observations[indices])
            all_dones.append(dones[indices])
            all_rewards.append(rewards[indices])

        # Concatenate the data after all batches processed
        data = (
            torch.cat(all_observations, dim=0),
            torch.cat(all_actions, dim=0),
            torch.cat(all_next_observations, dim=0),
            torch.cat(all_dones, dim=0),
            torch.cat(all_rewards, dim=0),
        )
        return ReplayBufferSamples(*tuple(data))


# Adapted from https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/tree.py
# TODO: Use https://github.com/rlcode/per/blob/master/prioritized_memory.py as reference
# TODO: Open SB3 issue: https://github.com/hill-a/stable-baselines/issues/751
class PrioritizedReplayBuffer(BaseBuffer):
    def __init__(self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        eps=1e-2,
        alpha=0.1,
        beta=0.1
    ):
        assert n_envs==1, "Current implementation does not support > 1 environements with tree object"
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage
        self.tree = SumTree(size=buffer_size)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # transition: state, action, reward, next_state, done
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # self.count = 0
        self.real_size = 0
        # self.size = buffer_size

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[Dict[str, Any]],
    ) -> None:

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.pos)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

        # update counters
        self.real_size = min(self.buffer_size, self.real_size + 1)

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, batch_size)
    
    def _get_samples(self, batch_inds: np.ndarray, batch_size: int) -> ReplayBufferSamples:
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        sample_idxs = np.array(sample_idxs)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self.observations[(sample_idxs + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[sample_idxs, env_indices, :]

        data = (
            self.observations[sample_idxs, env_indices, :],
            self.actions[sample_idxs, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[sample_idxs, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[sample_idxs, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data))), weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)


class PrioritizedCriticalReplayBuffer(BaseBuffer):
    def __init__(self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        eps=1e-2,
        alpha=0.1,
        beta=0.1
    ):
        assert n_envs==1, "Current implementation does not support > 1 environements with tree object"
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage
        self.tree = SumTree(size=buffer_size)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # transition: state, action, reward, next_state, done
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.real_size = 0

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[Dict[str, Any]],
    ) -> None:

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.pos)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

        # update counters
        self.real_size = min(self.buffer_size, self.real_size + 1)

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, batch_size)
    
    def _get_samples(self, batch_inds: np.ndarray, batch_size: int) -> ReplayBufferSamples:
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        sample_idxs = np.array(sample_idxs)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self.observations[(sample_idxs + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[sample_idxs, env_indices, :]

        data = (
            self.observations[sample_idxs, env_indices, :],
            self.actions[sample_idxs, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[sample_idxs, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[sample_idxs, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data))), weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

