import numpy as np
import torch
from scipy.stats import norm

from base import SimulationBase
from utils import SimpleReplayBuffer

class DRL(SimulationBase):
    """
    Kallus, Nathan, and Masatoshi Uehara. "Efficiently breaking the curse of horizon in off-policy 
    evaluation with double reinforcement learning." Operations Research (2022).
    """
    
    def __init__(self, env, n, horizon, eval_env=None, discount=0.9, device=None, seed=0):

        super().__init__(
            env=env, 
            n=n, 
            horizon=horizon, 
            discount=discount, 
            eval_env=eval_env
        )

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.gamma = discount
        self.ipw = False
        self.prob_lbound = 1. # default

    def estimate_omega(self, omega_estimator, target_policy, initial_state_sampler, estimate_omega_kwargs):
        if not hasattr(self, 'target_policy'): 
            self.target_policy = target_policy
        self.initial_state_sampler = initial_state_sampler

        self.omega_estimator = omega_estimator
        self.omega_estimator.masked_buffer = self.masked_buffer
        self.omega_estimator.burn_in = self.burn_in
        self.omega_estimator.propensity_pred = self.propensity_pred
        self.omega_estimator.estimate_omega(
            target_policy=target_policy, 
            initial_state_sampler=initial_state_sampler, 
            **estimate_omega_kwargs
        )
        if estimate_omega_kwargs.get('ipw', False):
            self.ipw = True
            self.prob_lbound = estimate_omega_kwargs['prob_lbound']

    def estimate_Q(self, Q_estimator, target_policy, estimate_Q_kwargs):
        if not hasattr(self, 'target_policy'): 
            self.target_policy = target_policy

        self.Q_estimator = Q_estimator
        self.Q_estimator.masked_buffer = self.masked_buffer
        self.Q_estimator.burn_in = self.burn_in
        self.Q_estimator._initial_obs = self._initial_obs
        self.Q_estimator.propensity_pred = self.propensity_pred
        self.Q_estimator.estimate_Q(
            target_policy=target_policy, 
            **estimate_Q_kwargs
        )
        if estimate_Q_kwargs.get('ipw', False):
            self.ipw = True
            self.prob_lbound = estimate_Q_kwargs['prob_lbound']

    def omega(self, states, actions):
        return self.omega_estimator.omega(states, actions)

    def Q(self, states, actions):
        return self.Q_estimator.Q(states, actions)

    def get_value(self, verbose=True):
        self.replay_buffer = self.omega_estimator.replay_buffer
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        rewards = self.replay_buffer.rewards
        next_states = self.replay_buffer.next_states
        initial_states = self.initial_state_sampler.initial_states # self._initial_obs
        if self.ipw:
            dropout_prob = self.replay_buffer.dropout_prob  # (NT,)
        else:
            dropout_prob = np.zeros_like(actions, dtype=float)
        inverse_wts = 1 / np.clip(a=1 - dropout_prob,
                                  a_min=self.prob_lbound,
                                  a_max=1).astype(float)
        total_T = self.omega_estimator.total_T_ipw

        est_omega = self.omega(states, actions)
        # est_omega = est_omega / np.mean(est_omega) # normalize

        est_Q = self.Q_estimator.Q(states, actions).squeeze()
        est_next_V = self.Q_estimator.V(next_states).squeeze()
        bellman_residual = rewards + self.gamma * est_next_V - est_Q
        Q_debias = 1 / (1 - self.gamma) * np.sum(est_omega * inverse_wts * bellman_residual) / total_T
        integrated_Q = np.mean(self.Q_estimator.V(initial_states))
        if verbose:
            print('omega: min = ', np.min(est_omega), 'max = ', np.max(est_omega))
            print('V_est(dm)', integrated_Q)
            print('V_est(mis)', np.sum(est_omega * inverse_wts * rewards) / (1 - self.gamma) / total_T)
            # print('v1', np.sum(est_omega * inverse_wts * rewards) / (1 - self.gamma) / total_T)
            # print('v2', np.sum(est_omega * inverse_wts * (self.gamma * est_next_V - est_Q)) / (1 - self.gamma) / total_T)
            # print('v3', integrated_Q)
        return Q_debias + integrated_Q

    def get_value_interval(self, alpha_list=[0.05]):
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        rewards = self.replay_buffer.rewards
        next_states = self.replay_buffer.next_states
        initial_states = self.initial_state_sampler.initial_states # self._initial_obs
        if self.ipw:
            dropout_prob = self.replay_buffer.dropout_prob  # (NT,)
        else:
            dropout_prob = np.zeros_like(actions, dtype=float)
        inverse_wts = 1 / np.clip(a=1 - dropout_prob,
                                  a_min=self.prob_lbound,
                                  a_max=1).astype(float)
        total_T = self.omega_estimator.total_T_ipw

        est_omega = self.omega(states, actions)
        # est_omega = est_omega / np.mean(est_omega) # normalize

        est_Q = self.Q_estimator.Q(states, actions).squeeze()
        est_next_V = self.Q_estimator.V(next_states).squeeze()
        bellman_residual = rewards + self.gamma * est_next_V - est_Q
        Q_debias = est_omega * inverse_wts * bellman_residual / (1 - self.gamma)
        integrated_Q = np.mean(self.Q_estimator.V(initial_states))
        V = np.sum(Q_debias) / total_T + integrated_Q
        V_int_sigma_sq = np.sum(Q_debias ** 2) / total_T
        std = (V_int_sigma_sq ** 0.5) / (total_T ** 0.5)
        inference_summary = {'value': V, 'std': std}

        lower_bound = {}
        upper_bound = {}
        for alpha in alpha_list:
            lower_bound[alpha] = V - norm.ppf(1 - alpha / 2) * std
            upper_bound[alpha] = V + norm.ppf(1 - alpha / 2) * std

        inference_summary.update({
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
        })
        return inference_summary