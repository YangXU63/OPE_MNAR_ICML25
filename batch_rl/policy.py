import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical


class CategoricalMLPPolicy(nn.Module):

    def __init__(self, state_dim, num_actions, hidden_sizes=[256, 256]):
        """
        Policy module for discrete action space.
        """
        super(CategoricalMLPPolicy, self).__init__()
        self._hidden_sizes = hidden_sizes
        self._state_dim = state_dim
        self.layers = nn.ModuleList()
        for prev_size, size in zip([state_dim] + hidden_sizes[:-1],
                                   hidden_sizes[1:]):
            hidden_layers = nn.Sequential()
            linear_layer = nn.Linear(in_features=prev_size, out_features=size)
            nn.init.xavier_normal_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)
            hidden_layers.add_module('linear', linear_layer)
            hidden_layers.add_module('non_linearity', nn.ReLU())
            self.layers.append(hidden_layers)

        self.output_layer = nn.Sequential()
        linear_layer = nn.Linear(in_features=hidden_sizes[-1],
                                 out_features=num_actions)
        self.output_layer.add_module('linear', linear_layer)
        self.output_layer.add_module('non_linearity', nn.Softmax(dim=1))

    def forward(self, state):
        x = state
        if self._hidden_sizes is not None:
            for layer in self.layers:
                x = layer(x)
        return self.output_layer(x)

    def get_actions(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(-1, self._state_dim)
            probs = self.forward(state)
        dists = Categorical(probs = probs)
        return dists.sample().detach().cpu().numpy()

    def sample_multiple(self, state, num_sample=10):
        state = torch.FloatTensor(state).reshape(-1, self._state_dim)
        probs = self.forward(state)
        dists = Categorical(probs=probs)
        z = dists.sample(sample_shape=torch.Size([num_sample])).T # (batch_size, num_sample)
        return None, z

    def log_pis(self, state, action):
        """Get log pi's for the model."""
        state = torch.FloatTensor(state).reshape((-1, self._state_dim))
        probs = self.forward(state)
        dists = Categorical(probs=probs)
        log_pis = dists.log_prob(value=action).sum(dim=-1)
        return log_pis # (batch_size,)

class TanhMLPPolicy(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 hidden_sizes=[400, 300]):
        super(TanhMLPPolicy, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], action_dim)
        self.max_action = max_action

    def forward(self, x, preval=False):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        pre_tanh_val = x
        x = self.max_action * torch.tanh(self.l3(x))
        if not preval:
            return x
        return x, pre_tanh_val

class TanhGaussianMLPPolicy(nn.Module):
    """A probabilistic actor which does regular stochastic mapping of actions from states"""
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 hidden_sizes=[400, 300]):
        super(TanhGaussianMLPPolicy, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.mean = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std = nn.Linear(hidden_sizes[1], action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)
        std_a = torch.exp(log_std_a)
        z = mean_a + std_a * torch.FloatTensor(
            np.random.normal(0, 1, size=(std_a.size())))
        return self.max_action * torch.tanh(z)

    def sample_multiple(self, state, num_sample=10):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)
        std_a = torch.exp(log_std_a)
        # This trick stabilizes learning (clipping gaussian to a smaller range)
        z = mean_a.unsqueeze(dim=1) + \
             std_a.unsqueeze(dim=1) * torch.FloatTensor(np.random.normal(0, 1, size=(std_a.size(0), num_sample, std_a.size(1)))).clamp(-0.5, 0.5)
        return self.max_action * torch.tanh(z), z

    def log_pis(self, state, action=None, raw_action=None):
        """Get log pi's for the model."""
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)
        std_a = torch.exp(log_std_a)
        normal_dist = torch.distributions.Normal(loc=mean_a,
                                                 scale=std_a,
                                                 validate_args=True)
        if raw_action is None:
            raw_action = atanh(action)
        else:
            action = torch.tanh(raw_action)
        log_normal = normal_dist.log_prob(raw_action)
        log_pis = log_normal.sum(-1)
        log_pis = log_pis - (1.0 - action**2).clamp(min=1e-6).log().sum(-1)
        return log_pis

class DiscreteQFArgmaxPolicy(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_sizes=[256, 256]):
        super(DiscreteQFArgmaxPolicy, self).__init__()
        self._hidden_sizes = hidden_sizes
        self._state_dim = state_dim
        self._num_actions = num_actions
        layers = []
        # hidden layers
        for prev_size, size in zip([state_dim] + hidden_sizes[:-1],
                                    hidden_sizes):
            linear_layer = nn.Linear(in_features=prev_size,
                                        out_features=size)
            nn.init.xavier_normal_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)
            layers.append(linear_layer)
            layers.append(nn.ReLU())
        # output layer
        output_layer = nn.Linear(in_features=hidden_sizes[-1],
                                    out_features=num_actions)
        layers.append(output_layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        Q_val = self.layers(state)
        return Q_val.argmax(dim=1, keepdim=True)

    def get_actions(self, state):
        with torch.no_grad():
            actions = self.forward(state)
        return actions.numpy()

    def load(self, filename):
        self.load_state_dict(torch.load(filename))