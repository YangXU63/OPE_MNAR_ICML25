"""
Code is modified from https://github.com/sfujim/BCQ/blob/master/discrete_BCQ/DQN.py
"""
import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from batch_rl.utils import moving_average

class QNetwork(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        hidden_sizes=[256, 256],
        hidden_nonlinearity=nn.ReLU(),
        hidden_w_init=nn.init.xavier_normal_,
        hidden_b_init=nn.init.zeros_,
        output_nonlinearity=None,
        output_bias=True,
        output_w_inits=nn.init.xavier_normal_,
        output_b_inits=nn.init.zeros_,
    ):
        super().__init__()

        self._hidden_sizes = hidden_sizes
        self._state_dim = input_dim
        self._num_actions = output_dim

        # build network structure
        layers = []

        # input layer
        input_layer = nn.Linear(in_features=self._state_dim,
                                out_features=hidden_sizes[0])
        nn.init.xavier_normal_(input_layer.weight)
        nn.init.zeros_(input_layer.bias)
        layers.append(input_layer)
        layers.append(hidden_nonlinearity)  # nn.LeakyReLU(0.01), nn.LeakyReLU(0.2)

        # hidden layers
        for prev_size, size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            linear_layer = nn.Linear(in_features=prev_size, out_features=size)
            hidden_w_init(linear_layer.weight)
            hidden_b_init(linear_layer.bias)
            layers.append(linear_layer)
            layers.append(hidden_nonlinearity)  # nn.LeakyReLU(0.01)

        # output layer
        output_layer = nn.Linear(in_features=hidden_sizes[-1],
                                 out_features=self._num_actions,
                                 bias=output_bias)
        output_w_inits(output_layer.weight)
        output_b_inits(output_layer.bias)
        layers.append(output_layer)
        if output_nonlinearity:
            layers.append(output_nonlinearity)

        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        return self.layers(state)

class DQN(object):
    def __init__(
        self, 
        num_actions,
        state_dim,
        device,
        discount=0.99,
        hidden_sizes=[256, 256],
        optimizer="Adam",
        optimizer_parameters={},
        polyak_target_update=False,
        target_update_frequency=8e3,
        tau=0.005,
        initial_eps = 1,
        end_eps = 0.001,
        eps_decay_period = 25e4,
        eval_eps=0.001):
        
        self.device = device

        # Determine network type
        self.Q = QNetwork(input_dim=state_dim, output_dim=num_actions, hidden_sizes=hidden_sizes).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)
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

        # Number of training iterations
        self.iterations = 0
        self.loss_history = []
    
    def select_action(self, state, eval=False):
        eps = self.eval_eps if eval \
            else max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(low=0,high=1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
                return int(self.Q(state).argmax(dim=1))
        else:
            return np.random.randint(self.num_actions)

    def select_actions(self, states, eval=False):
        eps = self.eval_eps if eval \
            else max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # select action according to policy with probability (1 - eps)
        # otherwise, select random action
        states = torch.FloatTensor(states).reshape(self.state_shape).to(self.device)
        random_actions = np.random.randint(self.num_actions, size=states.size(0))
        with torch.no_grad():
            opt_actions = self.Q(states).argmax(dim=1).numpy()
        return np.where(np.random.uniform(low=0,high=1, size=states.size(0)) > eps, opt_actions, random_actions)

    def train_once(self, replay_buffer, minibatch_size=None):
        if hasattr(replay_buffer,'priority_prob'):
            self.prioritized_replay = True
        else:
            self.prioritized_replay = False
        # Sample replay buffer
        if self.prioritized_replay:
            state, action, next_state, reward, not_done, idxs, imp_weights = replay_buffer.sample(batch_size=minibatch_size)
        else:
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size=minibatch_size)
            idxs, imp_weights = None, None

        # Compute the target Q value
        with torch.no_grad():
            target_Q = reward + not_done * self.discount * self.Q_target(next_state).max(dim=1, keepdim=True)[0]

        # Get current Q estimate
        current_Q = self.Q(state).gather(dim=1, index=action)

        # get absolute errors and update prioritized sampling probabilities
        abs_errors = torch.abs(current_Q - target_Q)
        if self.prioritized_replay:
            priority_prob = (abs_errors.data.numpy() + replay_buffer.per_epsilon) ** replay_buffer.per_alpha
            replay_buffer.update(idxs=idxs, priorities=priority_prob)

        # Compute Q loss, closely related to HuberLoss
        if imp_weights is None:
            # unweighted smooth L1 loss
            Q_loss = F.smooth_l1_loss(input=current_Q, target=target_Q)
        else:
            # weighted smooth L1 loss
            Q_loss = torch.mean(imp_weights * torch.where(abs_errors < 1, 0.5 * abs_errors ** 2, abs_errors - 0.5))

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()
        self.loss_history.append(Q_loss.detach().cpu().numpy().item())

    def train(self, replay_buffer, max_iters=1e6, minibatch_size=None, verbose=False):
        self.Q.train()
        for i in range(int(max_iters)):
            self.train_once(replay_buffer=replay_buffer, minibatch_size=minibatch_size)
            if verbose:
                if self.iterations % (max_iters // 10) == 0:
                    print(f'iteration {self.iterations}: loss {self.loss_history[-1]}')

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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
            Q_val = self.Q(states_tensor)
        opt_actions = Q_val.argmax(dim=1, keepdim=True)
        Q_val = Q_val.gather(dim=1, index=opt_actions).detach().cpu().numpy()
        return np.mean(Q_val)

    def plot_loss(self, smooth_window=1000):
        fig = plt.figure()
        _ = sns.lineplot(data=moving_average(np.array(self.loss_history).flatten(), smooth_window))
        return fig

class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 256]):
        super().__init__()

        self._hidden_sizes = hidden_sizes
        self._state_dim = input_dim
        self._num_actions = output_dim

        layers = []
        # hidden layers
        for prev_size, size in zip([self._state_dim] + hidden_sizes[:-1],
                                    hidden_sizes):
            linear_layer = nn.Linear(in_features=prev_size,
                                        out_features=size)
            nn.init.xavier_normal_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)
            layers.append(linear_layer)
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.LeakyReLU(0.01))
        self.layers = nn.Sequential(*layers)
        
        # output layer
        self.value_output_layer = nn.Linear(in_features=hidden_sizes[-1] // 2,
                                    out_features=1)
        self.adv_output_layer = nn.Linear(in_features=hidden_sizes[-1] - hidden_sizes[-1] // 2,
                                    out_features=self._num_actions)

    def forward(self, state):
        output = self.layers(state)
        value = self.value_output_layer(output[:,:self._hidden_sizes[-1] // 2])
        adv = self.adv_output_layer(output[:,self._hidden_sizes[-1] // 2:])
        avg_adv = torch.mean(adv, dim=1, keepdim=True)
        Q_value = value + adv - avg_adv
        return Q_value

    def select_action(self, state):
        with torch.no_grad():
            Q_value = self.forward(state)
        return torch.argmax(Q_value, dim=1).detach().numpy()

class DuelingDQN(object):
    def __init__(
        self, 
        num_actions,
        state_dim,
        device,
        discount=0.99,
        hidden_sizes=[256, 256],
        optimizer="Adam",
        optimizer_parameters={},
        polyak_target_update=False,
        target_update_frequency=8e3,
        tau=0.005,
        initial_eps = 1,
        end_eps = 0.001,
        eps_decay_period = 25e4,
        eval_eps=0.001):
        
        self.device = device

        # Determine network type
        self.Q = DuelingQNetwork(input_dim=state_dim, output_dim=num_actions, hidden_sizes=hidden_sizes).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)
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

        # Number of training iterations
        self.iterations = 0
        self.loss_history = []
    
    def select_action(self, state, eval=False):
        eps = self.eval_eps if eval \
            else max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(low=0,high=1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
                return int(self.Q(state).argmax(dim=1))
        else:
            return np.random.randint(self.num_actions)

    def select_actions(self, states, eval=False):
        eps = self.eval_eps if eval \
            else max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        states = torch.FloatTensor(states).reshape(self.state_shape).to(self.device)
        random_actions = np.random.randint(self.num_actions, size=states.size(0))
        with torch.no_grad():
            opt_actions = self.Q(states).argmax(dim=1).numpy()
        return np.where(np.random.uniform(low=0,high=1, size=states.size(0)) > eps, opt_actions, random_actions)

    def train_once(self, replay_buffer, minibatch_size=None):
        if hasattr(replay_buffer,'priority_prob'):
            self.prioritized_replay = True
        else:
            self.prioritized_replay = False
        # Sample replay buffer
        if self.prioritized_replay:
            state, action, next_state, reward, not_done, idxs, imp_weights = replay_buffer.sample(batch_size=minibatch_size)
        else:
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size=minibatch_size)
            idxs, imp_weights = None, None

        # Compute the target Q value
        with torch.no_grad():
            target_Q = reward + not_done * self.discount * self.Q_target(next_state).max(dim=1, keepdim=True)[0]

        # Get current Q estimate
        current_Q = self.Q(state).gather(dim=1, index=action)

        ## get absolute errors and update prioritized sampling probabilities
        abs_errors = torch.abs(current_Q - target_Q)
        if self.prioritized_replay:
            priority_prob = (abs_errors.data.numpy() + replay_buffer.per_epsilon) ** replay_buffer.per_alpha
            replay_buffer.update(idxs=idxs, priorities=priority_prob)

        # Compute Q loss, closely related to HuberLoss
        if imp_weights is None:
            ## unweighted smooth L1 loss
            Q_loss = F.smooth_l1_loss(input=current_Q, target=target_Q)
        else:
            ## weighted smooth L1 loss
            Q_loss = torch.mean(imp_weights * torch.where(abs_errors < 1, 0.5 * abs_errors ** 2, abs_errors - 0.5))

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()
        self.loss_history.append(Q_loss.detach().cpu().numpy().item())

    def train(self, replay_buffer, max_iters=1e6, minibatch_size=None, verbose=False):
        self.Q.train()
        for i in range(int(max_iters)):
            self.train_once(replay_buffer=replay_buffer, minibatch_size=minibatch_size)
            if verbose:
                if self.iterations % (max_iters // 10) == 0:
                    print(f'iteration {self.iterations}: loss {self.loss_history[-1]}')

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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
            Q_val = self.Q(states_tensor)
        opt_actions = Q_val.argmax(dim=1, keepdim=True)
        Q_val = Q_val.gather(dim=1, index=opt_actions).detach().cpu().numpy()
        return np.mean(Q_val)

    def plot_loss(self, smooth_window=1000):
        fig = plt.figure()
        _ = sns.lineplot(data=moving_average(np.array(self.loss_history).flatten(), smooth_window))
        return fig
    
    
    
