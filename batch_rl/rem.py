"""
Code is modified from https://github.com/google-research/batch_rl
"""
import os
import sys
import copy
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from batch_rl.dqn import QNetwork
from batch_rl.utils import moving_average

MultiNetworkNetworkType = collections.namedtuple(
    'multi_network_dqn_network',
    ['q_networks', 'unordered_q_networks', 'q_values'])


def combine_q_functions(q_functions, transform_strategy, **kwargs):
    """
    Utility function for combining multiple Q functions.

    Parameters
    ----------
    q_functions: Multiple Q-functions concatenated.
    transform_strategy (str) : Possible options include (1) 'identity' for no
        transformation (2) 'stochastic' for random convex combination.
    **kwargs : keyword arguments passed to `transform_matrix`, the matrix for transforming the Q-values if the passed
        `transform_strategy` is `stochastic`.

    Returns
    ----------
    q_functions: Modified Q-functions.
    q_values: Q-values based on combining the multiple heads.
    """
    # Create q_values before reordering the heads for training
    q_values = torch.mean(q_functions, dim=-1)

    if transform_strategy not in ['stochastic', 'identity']:
        raise ValueError(
            '{} is not a valid reordering strategy'.format(transform_strategy))
    if transform_strategy == 'stochastic':
        left_stochastic_matrix = kwargs.get('transform_matrix')
        if left_stochastic_matrix is None:
            raise ValueError('None value provided for stochastic matrix')
        q_functions = torch.tensordot(q_functions,
                                      left_stochastic_matrix,
                                      dims=[[2], [0]])
    return q_functions, q_values


def random_stochastic_matrix(dim, num_cols=None, dtype=torch.float32):
    """Generates a random left stochastic matrix."""
    mat_shape = (dim, dim) if num_cols is None else (dim, num_cols)
    mat = torch.rand(size=mat_shape, dtype=dtype)
    mat /= torch.linalg.norm(mat, ord=1, dim=0, keepdim=True)
    return mat


class MulitNetworkQNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_networks,
                 hidden_sizes=[256, 256],
                 transform_strategy='stochastic',
                 **kwargs):
        """
        Parameters
		----------
        input_dim (int) : input dimension.
        output_dim (int) : output dimension.
        hidden_sizes (list) : sizes of hidden layers.
        num_networks (int) : number of separate Q-networks.
        transform_strategy (str) : Possible options include 
            (1) 'identity' for no transformation (Ensemble-DQN) 
            (2) 'stochastic' for random convex combination (REM).
        kwargs: keyword arguments passed to `transform_matrix`, the matrix for transforming the Q-values if only
            the passed `transform_strategy` is `stochastic`.
        """
        super().__init__()
        self._state_dim = input_dim
        self._num_actions = output_dim
        self._num_networks = num_networks
        self._transform_strategy = transform_strategy
        self._kwargs = kwargs
        # Create multiple Q-networks
        # self._q_networks = []
        self._q_networks = nn.ModuleList()
        for i in range(self._num_networks):
            q_net = QNetwork(input_dim=self._state_dim,
                             output_dim=self._num_actions,
                             hidden_sizes=hidden_sizes)
            self._q_networks.append(q_net)

    def forward(self, state):
        unordered_q_networks = [network(state) for network in self._q_networks]
        unordered_q_networks = torch.stack(unordered_q_networks, dim=-1)
        q_networks, q_values = combine_q_functions(unordered_q_networks,
                                                   self._transform_strategy,
                                                   **self._kwargs)
        return MultiNetworkNetworkType(q_networks, unordered_q_networks,
                                       q_values)


class REM(object):
    def __init__(self,
                 state_dim,
                 num_actions,
                 num_networks,
                 device,
                 transform_strategy='identity',
                 num_convex_combinations=1,
                 hidden_sizes=[256, 256],
                 discount=0.99,
                 optimizer="Adam",
                 optimizer_parameters={},
                 use_deep_exploration=False,
                 polyak_target_update=False,
                 target_update_frequency=8e3,
                 tau=0.005,
                 initial_eps=1,
                 end_eps=0.001,
                 eps_decay_period=25e4,
                 eval_eps=0.001):
        """
        Random Ensemble Mixture (REM).

        Parameters
        ----------
        num_actions (int) : Number of actions the agent can take at any state.
        num_networks (int) : Number of different Q-functions.
        device (torch.device) : Device on which the agent's graph is executed.
        transform_strategy (str) : Possible options include (1) 'identity' for no
            transformation (2) 'stochastic' for random convex combination.
        num_convex_combinations (int) : If transform_strategy is 'stochastic', then this argument specifies the number of random
            convex combinations to be created. If None, `num_heads` convex combinations are created.
        discount (float) : Discount factor with the usual RL meaning.
        optimizer (str) : Optimizer for training the value function.
        use_deep_exploration (bool) : Adaptation of Bootstrapped DQN for REM exploration.
        """

        self.device = device
        self.num_actions = num_actions
        self.num_networks = num_networks
        self._q_networks_transform = None
        self._num_convex_combinations = num_convex_combinations
        self.transform_strategy = transform_strategy
        self.use_deep_exploration = use_deep_exploration

        kwargs = {}
        if self._q_networks_transform is None:
            if self.transform_strategy == 'stochastic':
                self._q_networks_transform = random_stochastic_matrix(
                    dim=self.num_networks,
                    num_cols=self._num_convex_combinations)
        if self._q_networks_transform is not None:
            kwargs.update({'transform_matrix': self._q_networks_transform})

        self.Q = MulitNetworkQNetwork(
            input_dim=state_dim,
            output_dim=self.num_actions,
            num_networks=self.num_networks,
            hidden_sizes=hidden_sizes,
            transform_strategy=self.transform_strategy,
            **kwargs)
        self.Q_target = copy.deepcopy(self.Q)
        # self.Q_optimizer = [getattr(torch.optim, optimizer)(self.Q._q_networks[i].parameters(), **optimizer_parameters) for i in range(self.num_networks)]
        self.Q_optimizer = getattr(torch.optim,
                                   optimizer)(self.Q.parameters(),
                                              **optimizer_parameters)
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

        # Number of training iterations
        self.iterations = 0
        self.loss_history = []

    def select_action(self, state, eval=False):
        eps = self.eval_eps if eval else max(
            self.slope * self.iterations + self.initial_eps, self.end_eps)
        if np.random.uniform(low=0, high=1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).reshape(self.state_shape).to(
                    self.device)
                Q_outputs = self.Q(state)
                self._q_argmax_eval = Q_outputs.q_values.argmax(dim=1)[0]
                if eval:
                    self._q_argmax = self._q_argmax_eval
                else:
                    if self.use_deep_exploration:
                        if self.transform_strategy.endswith('stochastic'):
                            q_transform = random_stochastic_matrix(
                                self.num_networks, num_cols=1)
                            episode_q_function = torch.tensordot(
                                Q_outputs.unordered_q_networks,
                                q_transform,
                                dims=[[2], [0]])
                            self._q_argmax_train = torch.argmax(
                                episode_q_function[:, :, 0], dim=1)[0]
                        elif self.transform_strategy == 'identity':
                            q_function_index = torch.randint(
                                low=0,
                                high=self.num_networks,
                                size=1,
                                dtype=torch.int32)
                            q_function = Q_outputs.unordered_q_networks[:, :,
                                                                        q_function_index]
                            self._q_argmax_train = torch.argmax(q_function,
                                                                dim=1)[0]
                    else:
                        self._q_argmax_train = self._q_argmax_eval
                    self._q_argmax = self._q_argmax_train
            return int(self._q_argmax)
        else:
            return np.random.randint(self.num_actions)

    def select_actions(self, states, eval=False):
        eps = self.eval_eps if eval \
            else max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        random_actions = np.random.randint(self.num_actions,
                                           size=states.size(0))
        with torch.no_grad():
            states = torch.FloatTensor(states).reshape(self.state_shape).to(
                self.device)
            Q_outputs = self.Q(states)
            self._q_argmax_eval = Q_outputs.q_values.argmax(dim=1)
            if eval:
                opt_actions = self._q_argmax_eval
            else:
                if self.use_deep_exploration:
                    if self.transform_strategy.endswith('stochastic'):
                        q_transform = random_stochastic_matrix(
                            self.num_networks, num_cols=1)
                        q_function = torch.tensordot(
                            Q_outputs.unordered_q_networks,
                            q_transform,
                            dims=[[2], [0]])
                        self._q_argmax_train = torch.argmax(q_function[:, :,
                                                                       0],
                                                            dim=1)
                    elif self.transform_strategy == 'identity':
                        q_function_index = torch.randint(
                            low=0,
                            high=self.num_networks,
                            size=states.size(0),
                            dtype=torch.int32)
                        q_function = Q_outputs.unordered_q_networks.gather(
                            dim=2, index=q_function_index)
                        self._q_argmax_train = torch.argmax(q_function, dim=1)
                else:
                    self._q_argmax_train = self._q_argmax_eval
                opt_actions = self._q_argmax_train
        return np.where(
            np.random.uniform(low=0, high=1, size=states.size(0)) > eps,
            opt_actions, random_actions)

    def train_once(self, replay_buffer, minibatch_size=None):
        if hasattr(replay_buffer,'priority_prob'):
            self.prioritized_replay = True
        else:
            self.prioritized_replay = False
        # Sample replay buffer
        # state, action, next_state, reward, not_done = replay_buffer.sample(
        #     batch_size=minibatch_size)
        if self.prioritized_replay:
            state, action, next_state, reward, not_done, idxs, imp_weights = replay_buffer.sample(batch_size=minibatch_size)
        else:
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size=minibatch_size)
            idxs, imp_weights = None, None

        # Compute the target Q value
        with torch.no_grad():
            next_target_outputs = self.Q_target(next_state).q_networks
            # next_qt_max = next_target_outputs.max(dim=1, keepdim=True)[0]
            next_qt_max = next_target_outputs.max(dim=1)[0]
            target_Q = reward + not_done * self.discount * next_qt_max

        # Get current Q estimate
        Q_val = self.Q(state).q_networks
        current_Q = Q_val.gather(
            dim=1,
            index=action.unsqueeze(dim=-1).repeat_interleave(
                repeats=Q_val.size(-1), dim=-1)).squeeze(dim=1)

        # Compute Q loss
        Q_loss = F.smooth_l1_loss(input=current_Q, target=target_Q)

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()
        self.loss_history.append(Q_loss.detach().cpu().numpy().item())

    def train(self,
              replay_buffer,
              max_iters=1e6,
              minibatch_size=None,
              verbose=False):
        self.Q.train()
        for _ in range(max_iters):
            self.train_once(replay_buffer=replay_buffer,
                            minibatch_size=minibatch_size)
            if verbose:
                if self.iterations % (max_iters // 10) == 0:
                    print(
                        f'iteration {self.iterations}: loss {self.loss_history[-1]}'
                    )

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
            Q_val = self.Q(states_tensor).q_values
        opt_actions = Q_val.argmax(dim=1, keepdim=True)
        Q_val = Q_val.gather(dim=1, index=opt_actions).detach().cpu().numpy()
        return np.mean(Q_val)

    def plot_loss(self, smooth_window=1000):
        fig = plt.figure()
        _ = sns.lineplot(data=moving_average(
            np.array(self.loss_history).flatten(), smooth_window))
        return fig
