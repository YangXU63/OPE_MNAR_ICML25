"""
Code is modified from https://github.com/sfujim/BCQ/blob/master/discrete_BCQ/utils.py
"""

import numpy as np
import torch

def moving_average(x, window):
    return np.convolve(x, np.ones(window), 'valid') / window

class ReplayBuffer(object):
    def __init__(
            self,
            state_dim,
            buffer_size,
            device):
        self.max_size = int(buffer_size)
        self.device = device

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, 1))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)
        self.initial_state.append()

    def add_episode(self, states, actions, next_states, rewards, dones):
        size = states.shape[0]
        self.state[self.ptr:(self.ptr + size)] = states
        self.action[self.ptr:(self.ptr + size)] = actions
        self.next_state[self.ptr:(self.ptr + size)] = next_states
        self.reward[self.ptr:(self.ptr + size)] = rewards
        self.not_done[self.ptr:(self.ptr + size)] = 1. - dones

        self.ptr = (self.ptr + size) % self.max_size
        self.crt_size = min(self.crt_size + size, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(low=0, high=self.crt_size, size=batch_size)
        return (torch.FloatTensor(self.state[ind]).to(self.device),
                torch.LongTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device))

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
        np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
        np.save(f"{save_folder}_next_state.npy",
                self.next_state[:self.crt_size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.state[:self.crt_size] = np.load(
            f"{save_folder}_state.npy")[:self.crt_size]
        self.action[:self.crt_size] = np.load(
            f"{save_folder}_action.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(
            f"{save_folder}_next_state.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.not_done[:self.crt_size] = np.load(
            f"{save_folder}_not_done.npy")[:self.crt_size]

        print(f"Replay Buffer loaded with {self.crt_size} elements.")


class ReplayBufferPER(object):
    per_alpha = 0.6
    per_epsilon = 0.01
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(
            self,
            state_dim,
            buffer_size,
            device):
        self.max_size = int(buffer_size)
        self.device = device

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, 1))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))
        self.priority_prob = np.ones((self.max_size, 1))

    def add(self, state, action, next_state, reward, done, priority=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        if priority is None:
            priority = 1
        self.priority_prob[self.ptr] = priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)
        self.initial_state.append()

    def add_episode(self, states, actions, next_states, rewards, dones, priorities=None):
        size = states.shape[0]
        self.state[self.ptr:(self.ptr + size)] = states
        self.action[self.ptr:(self.ptr + size)] = actions
        self.next_state[self.ptr:(self.ptr + size)] = next_states
        self.reward[self.ptr:(self.ptr + size)] = rewards
        self.not_done[self.ptr:(self.ptr + size)] = 1. - dones
        if priorities is None:
            priorities = np.ones(shape=(size, 1))
        self.priority_prob[self.ptr:(self.ptr + size)] = priorities

        self.ptr = (self.ptr + size) % self.max_size
        self.crt_size = min(self.crt_size + size, self.max_size)

    def sample(self, batch_size):
        normalized_priority_prob = self.priority_prob[:self.crt_size] / np.sum(self.priority_prob[:self.crt_size])
        ind = np.random.choice(a=np.arange(self.crt_size), size=batch_size, p=normalized_priority_prob.squeeze())
        priorities = self.priority_prob[ind]
        sampling_probabilities = priorities / sum(self.priority_prob)
        imp_weights = np.power(self.crt_size * sampling_probabilities, - self.beta)
        imp_weights /= imp_weights.max()
        return (torch.FloatTensor(self.state[ind]).to(self.device),
                torch.LongTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                torch.FloatTensor(ind).to(self.device),
                torch.FloatTensor(imp_weights).to(self.device))

    def update(self, idxs, priorities):  
        """update PER sampling probabilities"""    
        idxs = idxs.numpy().astype('int')
        self.priority_prob[idxs] = priorities

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
        np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
        np.save(f"{save_folder}_next_state.npy",
                self.next_state[:self.crt_size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
        np.save(f"{save_folder}_priority_prob.npy", self.priority_prob[:self.crt_size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.state[:self.crt_size] = np.load(
            f"{save_folder}_state.npy")[:self.crt_size]
        self.action[:self.crt_size] = np.load(
            f"{save_folder}_action.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(
            f"{save_folder}_next_state.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.not_done[:self.crt_size] = np.load(
            f"{save_folder}_not_done.npy")[:self.crt_size]
        self.priority_prob[:self.crt_size] = np.load(
            f"{save_folder}_priority_prob.npy")[:self.crt_size]

        print(f"Replay Buffer loaded with {self.crt_size} elements.")
