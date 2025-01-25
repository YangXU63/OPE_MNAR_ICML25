import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from batch_rl.policy import CategoricalMLPPolicy

class BC(object):
    def __init__(self,
                 num_actions,
                 state_dim,
                 device,
                 discount=0.99,
                 hidden_sizes=[256, 256],
                 optimizer="Adam",
                 optimizer_parameters={}):

        self.device = device
        # Determine network type
        self.policy = CategoricalMLPPolicy(state_dim=state_dim, num_actions=num_actions, hidden_sizes=hidden_sizes).to(self.device)
        
        self.policy_optimizer = getattr(torch.optim, optimizer)(self.policy.parameters(), **optimizer_parameters)
        self.discount = discount
        self.num_actions = num_actions
        self.state_shape = (-1, state_dim)

        self.iterations = 0

    def select_actions(self, states, deterministic=False):
        with torch.no_grad():
            states = torch.FloatTensor(states).reshape(self.state_shape).to(self.device)
            probs = self.policy(states)
        if deterministic:
            actions = probs.argmax(dim=1).detach().numpy()
        else:
            actions = self.policy.get_actions(states=states)
        return actions

    def select_action(self, state, deterministic=False):
        return self.select_actions(states=state, deterministic=deterministic).squeeze()[0]

    def train_once(self, replay_buffer, minibatch_size=None):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size=minibatch_size)
        policy_actions = self.policy(state)
        
        loss = F.nll_loss(input=policy_actions.log(), target=action)

        # Optimize the policy
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        self.iterations += 1

    def train(self, replay_buffer, max_iters=1e6, minibatch_size=None):
        for _ in range(max_iters):
            self.train_once(replay_buffer=replay_buffer, minibatch_size=minibatch_size)

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename + "_policy")
        torch.save(self.policy_optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename + "_policy"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_optimizer"))
