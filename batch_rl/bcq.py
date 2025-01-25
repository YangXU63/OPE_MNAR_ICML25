"""
Code is modified from https://github.com/sfujim/BCQ/blob/master/discrete_BCQ/discrete_BCQ.py
"""

import os
import sys
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from batch_rl.utils import ReplayBuffer, moving_average


class BehaviorQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 256]):
        super().__init__()

        self._hidden_sizes = hidden_sizes
        self._state_dim = input_dim
        self._num_actions = output_dim

        # Q-funtion
        Q_layers = []
        # hidden layers
        for prev_size, size in zip([self._state_dim] + hidden_sizes[:-1], hidden_sizes):
            linear_layer = nn.Linear(in_features=prev_size,
                                        out_features=size)
            nn.init.xavier_normal_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)
            Q_layers.append(linear_layer)
            Q_layers.append(nn.ReLU())
        # output layer
        output_layer = nn.Linear(in_features=hidden_sizes[-1],
                                    out_features=self._num_actions)
        Q_layers.append(output_layer)
        self.Q_layers = nn.Sequential(*Q_layers)
        
        # generative model
        G_layers = []
        # hidden layers
        for prev_size, size in zip([self._state_dim] + hidden_sizes[:-1],
                                    hidden_sizes):
            linear_layer = nn.Linear(in_features=prev_size,
                                        out_features=size)
            nn.init.xavier_normal_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)
            G_layers.append(linear_layer)
            G_layers.append(nn.ReLU())
        # output layer
        output_layer = nn.Linear(in_features=hidden_sizes[-1],
                                    out_features=self._num_actions)
        G_layers.append(output_layer)
        self.G_layers = nn.Sequential(*G_layers)

    def forward(self, state):
        q = self.Q_layers(state)
        bc = self.G_layers(state)
        return q, F.log_softmax(bc, dim=1), bc


class discrete_BCQ(object):
    def __init__(
        self,
        num_actions,
        state_dim,
        device,
        BCQ_threshold=0.3,
        hidden_sizes=[256,256],
        discount=0.99,
        optimizer="Adam",
        optimizer_parameters={},
        polyak_target_update=False,
        target_update_frequency=8e3,
        tau=0.005,
        initial_eps=1,
        end_eps=0.001,
        eps_decay_period=25e4,
        eval_eps=0.001):
        
        self.device = device

        # Determine network type
        self.Q = BehaviorQNetwork(input_dim=state_dim, output_dim=num_actions, hidden_sizes=hidden_sizes).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim,optimizer)(self.Q.parameters(),**optimizer_parameters)
        self.discount = discount

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Decay for eps
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        # Evaluation hyper-parameters
        self.state_shape = (-1, state_dim)
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        # Threshold for "unlikely" actions
        self.threshold = BCQ_threshold

        # Number of training iterations
        self.iterations = 0
        self.loss_history = []

    def select_action(self, state, eval=False):
        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(low=0, high=1) > self.eval_eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).reshape(self.state_shape).to(
                    self.device)
                q, log_prob, bc = self.Q(state)
                prob = log_prob.exp()
                prob = (prob / prob.max(dim=1, keepdim=True)[0] >
                        self.threshold).float()
                # Use large negative number to mask actions from argmax
                return int((prob * q + (1. - prob) * (-1e8)).argmax(dim=1))
        else:
            return np.random.randint(self.num_actions)

    def select_actions(self, states, eval=False):
        random_actions = np.random.randint(self.num_actions, size=states.size(0))
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(self.state_shape).to(
                self.device)
            q, log_prob, bc = self.Q(state)
            prob = log_prob.exp()
            prob = (prob / prob.max(dim=1, keepdim=True)[0] >
                    self.threshold).float()
            opt_actions = (prob * q + (1. - prob) * (-1e8)).argmax(dim=1)
        return np.where(
            np.random.uniform(low=0, high=1, size=states.size(0)) >
            self.eval_eps, opt_actions, random_actions)

    def train_once(self, replay_buffer, minibatch_size=None):
        if hasattr(replay_buffer,'priority_prob'):
            self.prioritized_replay = True
        else:
            self.prioritized_replay = False
        # Sample replay buffer
        # state, action, next_state, reward, not_done = replay_buffer.sample(batch_size=minibatch_size)
        if self.prioritized_replay:
            state, action, next_state, reward, not_done, idxs, imp_weights = replay_buffer.sample(batch_size=minibatch_size)
        else:
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size=minibatch_size)
            idxs, imp_weights = None, None
        # Compute the target Q value
        with torch.no_grad():
            q, log_prob, bc = self.Q(next_state)
            prob = log_prob.exp()
            candidate_action = (prob / prob.max(dim=1, keepdim=True)[0] >
                    self.threshold).float()
            # Use large negative number to mask actions from argmax
            next_action = (candidate_action * q + (1 - candidate_action) * (-1e8)).argmax(dim=1, keepdim=True)
            q, log_prob, bc = self.Q_target(next_state)
            target_Q = reward + not_done * self.discount * q.gather(
                dim=1, index=next_action).reshape(-1, 1)
        # Get current Q estimate
        current_Q, log_prob, bc = self.Q(state)
        current_Q = current_Q.gather(dim=1, index=action)
        # Compute Q loss
        q_loss = F.smooth_l1_loss(input=current_Q, target=target_Q)
        i_loss = F.nll_loss(input=log_prob, target=action.reshape(-1))
        Q_loss = q_loss + i_loss + 1e-3 * bc.pow(2).mean()
        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()
        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.loss_history.append(Q_loss.detach().cpu().numpy())
        self.maybe_update_target()

    def train(self, replay_buffer, max_iters=1e6, minibatch_size=None, verbose=False):
        self.Q.train()
        for _ in range(max_iters):
            self.train_once(replay_buffer=replay_buffer, minibatch_size=minibatch_size)
            if verbose:
                if self.iterations % (max_iters // 10) == 0:
                    print(f'iteration {self.iterations}: loss {self.loss_history[-1]}')
   
    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(),
                                        self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data +
                                    (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename):
        torch.save(self.Q.state_dict(), filename + "_Q")
        torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.Q.load_state_dict(torch.load(filename + "_Q"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))

    def evaluate(self, initial_states):
        self.Q.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(initial_states).to(self.device)
            Q_val, _, _ = self.Q(states_tensor)
        opt_actions = Q_val.argmax(dim=1, keepdim=True)
        Q_val = Q_val.gather(dim=1, index=opt_actions).detach().cpu().numpy()
        return np.mean(Q_val)

    def plot_loss(self, smooth_window=1000):
        fig = plt.figure()
        _ = sns.lineplot(data=moving_average(np.array(self.loss_history).flatten(), smooth_window))
        return fig
