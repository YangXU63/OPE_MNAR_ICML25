"""Note: currently the code is only compatible with continues state space and discrete action space"""

import os
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from base import SimulationBase
from density import (StateActionVisitationRatio,
                     StateActionVisitationRatioSpline,
                     StateActionVisitationRatioExpoLinear,
                     StateActionVisitationModel,
                     Square, Exp, Log1pexp)
from utils import SimpleReplayBuffer, normcdf, iden, MinMaxScaler
from batch_rl.dqn import QNetwork

__all__ = ['MWL', 'DualDice', 'NeuralDice']


class MWL(SimulationBase):
    """
    Uehara, Masatoshi, Jiawei Huang, and Nan Jiang. "Minimax weight and q-function learning for off-policy evaluation." 
    International Conference on Machine Learning. PMLR, 2020.
    """

    def __init__(self,
                 env,
                 n,
                 horizon,
                 eval_env=None,
                 discount=0.9,
                 device=torch.device('cpu'),
                 seed=0):

        super().__init__(env=env,
                         n=n,
                         horizon=horizon,
                         discount=discount,
                         eval_env=eval_env)

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.gamma = discount
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def estimate_omega(self,
                       target_policy,
                       initial_state_sampler,
                       omega_func_class='nn',
                       Q_func_class='rkhs',
                       hidden_sizes=[64, 64],
                       action_dim=1,
                       separate_action=False,
                       max_iter=100,
                       batch_size=32,
                       lr=0.0002,
                       ipw=False,
                       estimate_missing_prob=True,
                       prob_lbound=1e-3,
                       scaler=None,
                       print_freq=20,
                       patience=5,
                       verbose=False,
                       **kwargs):

        if not estimate_missing_prob:
            self.replay_buffer = SimpleReplayBuffer(trajs=self.masked_buffer, max_T=self.max_T - self.burn_in, 
                                                seed=self.seed)
        else:
            assert hasattr(
                self, 'propensity_pred'
            ), 'please call function self.estimate_missing_prob() first'
            self.replay_buffer = SimpleReplayBuffer(trajs=self.masked_buffer, max_T=self.max_T - self.burn_in, 
                                                prop_info=self.propensity_pred,
                                                seed=self.seed)

        self.target_policy = target_policy
        self.omega_func_class = omega_func_class
        self.Q_func_class = Q_func_class
        self.ipw = ipw
        self.prob_lbound = prob_lbound
        self.separate_action = separate_action
        self.scaler = scaler
        self.verbose = verbose

        if self.separate_action:
            action_dim = 0

        # curr_time = time.time()
        if self.omega_func_class == 'nn' and self.Q_func_class == 'rkhs':
            omega_func = StateActionVisitationRatio(
                replay_buffer=self.replay_buffer,
                initial_state_sampler=initial_state_sampler,
                Q_func_class=self.Q_func_class,
                discount=self.gamma,
                # action_range=self.action_range,
                hidden_sizes=hidden_sizes,
                num_actions=self.num_actions,
                action_dim=action_dim,
                separate_action=self.separate_action,
                lr=lr,
                scaler=scaler,
                device=self.device)

            omega_func.fit(
                target_policy=target_policy,
                batch_size=batch_size,
                max_iter=max_iter,
                ipw=ipw,
                prob_lbound=prob_lbound,
                print_freq=print_freq,
                patience=patience,
                # rep_loss=rep_loss
            )
            # omega_func = StateActionVisitationRatio(
            #     replay_buffer=self.replay_buffer,
            #     initial_state_sampler=initial_state_sampler,
            #     discount=self.gamma,
            #     hidden_sizes=hidden_sizes,
            #     num_actions=self.num_actions,
            #     # q_lr=1e-3,
            #     # omega_lr=1e-3,
            #     scaler=scaler,
            #     device=self.device,
            #     **kwargs
            # )
            # omega_func.fit(
            #     target_policy=target_policy,
            #     batch_size=batch_size,
            #     max_iter=max_iter,
            #     ipw=ipw,
            #     prob_lbound=prob_lbound,
            #     print_freq=print_freq,
            # )
            self.omega_func = omega_func
            # self.omega_model = omega_func.model
        elif self.omega_func_class == 'spline':
            omega_func = StateActionVisitationRatioSpline(
                replay_buffer=self.replay_buffer,
                initial_state_sampler=initial_state_sampler,
                Q_func_class=self.Q_func_class,
                discount=self.gamma,
                scaler=scaler,
                num_actions=self.num_actions)

            omega_func.fit(target_policy=target_policy,
                           ipw=ipw,
                           prob_lbound=prob_lbound,
                           **kwargs)

            self.omega_func = omega_func
        elif self.omega_func_class == 'expo_linear':
            omega_func = StateActionVisitationRatioExpoLinear(
                replay_buffer=self.replay_buffer,
                initial_state_sampler=initial_state_sampler,
                Q_func_class=self.Q_func_class,
                discount=self.gamma,
                scaler=scaler,
                num_actions=self.num_actions)

            omega_func.fit(target_policy=target_policy,
                           ipw=ipw,
                           prob_lbound=prob_lbound,
                           batch_size=batch_size,
                           max_iter=max_iter,
                           lr=lr,
                           print_freq=print_freq,
                           patience=patience,
                           **kwargs)

            self.omega_func = omega_func
        else:
            raise NotImplementedError
        
        self.total_T_ipw = self.omega_func.total_T_ipw

    def omega(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        inputs = np.hstack([states, actions[:, np.newaxis]])
        omega_values = self.omega_func.omega_prediction(inputs=inputs).squeeze()
        return omega_values

    def get_value(self, verbose=True):
        # state, action, reward, next_state = [np.array([item[i] for traj in self.trajs for item in traj]) for i in range(4)]
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        rewards = self.replay_buffer.rewards  # (NT,)
        if self.ipw:
            dropout_prob = self.replay_buffer.dropout_prob  # (NT,)
        else:
            dropout_prob = np.zeros_like(actions)
        inverse_wts = 1 / np.clip(a=1 - dropout_prob,
                                  a_min=self.prob_lbound,
                                  a_max=1).astype(float)  # (NT,)
        # next_state = self.replay_buffer.next_states
        est_omega = self.omega(states, actions)  #  (NT,)

        ## sanity check for spline
        # xi_mat = self.omega_func._Xi(S=states, A=actions)
        # mat2 = np.sum(xi_mat * (inverse_wts * rewards).reshape(-1,1), axis=0) / self.omega_func.total_T_ipw
        # V_int_IS = 1 / (1 - self.gamma) * np.matmul(mat2.T, self.omega_func.est_alpha)

        # est_omega = np.clip(est_omega, a_min=0, a_max=None) # clip, avoid negative values
        # est_omega = est_omega / np.mean(est_omega)  # normalize, move it into self.omega_func class

        # if self.func_class == 'spline':
        #     V_int_IS = 1 / (1 - self.gamma) * np.sum(est_omega * rewards * inverse_wts) / self.omega_func.total_T_ipw
        # else:
        #     V_int_IS = 1 / (1 - self.gamma) * np.mean(est_omega * rewards * inverse_wts)
        V_int_IS = 1 / (1 - self.gamma) * np.sum(est_omega * rewards * inverse_wts) / self.omega_func.total_T_ipw # key!
        
        if verbose: # self.verbose
            print('omega: min = ', np.min(est_omega), 'max = ', np.max(est_omega))
            print("value_est = {:.3f}\n".format(V_int_IS))

        return V_int_IS


class DualDice(SimulationBase):
    """
    Nachum, Ofir, et al. "Dualdice: Behavior-agnostic estimation of discounted stationary distribution corrections." 
    Advances in Neural Information Processing Systems 32 (2019).

    Official implementation: https://github.com/google-research/dice_rl/blob/master/estimators/neural_dual_dice.py
    """

    def __init__(self,
                 env,
                 n,
                 horizon,
                 eval_env=None,
                 discount=0.9,
                 device=torch.device('cpu'),
                 seed=0):

        super().__init__(env=env,
                         n=n,
                         horizon=horizon,
                         discount=discount,
                         eval_env=eval_env)

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.gamma = discount

        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def estimate_omega(
        self,
        target_policy,
        initial_state_sampler,
        nu_network_kwargs,
        nu_learning_rate,
        zeta_network_kwargs,
        zeta_learning_rate,
        zeta_pos: bool = False,
        solve_for_state_action_ratio: bool = True,
        ipw: bool = False,
        estimate_missing_prob: bool = True,
        prob_lbound: float = 1e-3,
        max_iter: int = 10000,
        batch_size: int = 1024,
        f_exponent: float = 2,
        primal_form: bool = False,
        scaler: str = "Identity",
        print_freq: int = 50,
        verbose: bool = False,
    ):
        """
        Estimate the state-action visitation ratio.

        Parameters
        ----------
        target_policy: DiscretePolicy
            The target policy to evaluate
        initial_state_sampler: InitialStateSampler
            Sampler for initial states and actions.
        nu_network_kwargs: dict
            Parameters of the nu-value network. Default {'hidden_sizes': [64, 64], 'hidden_nonlinearity': nn.ReLU(), 
            'hidden_w_init': nn.init.xavier_uniform_, 'output_w_inits': nn.init.xavier_uniform_}
        nu_learning_rate: float, 0.0001
            Learning rate for nu. 
        zeta_network_kwargs: dict
            Parameters of the zeta-value network. 
            Default {'hidden_sizes': [64, 64], 
            'hidden_nonlinearity': nn.ReLU(), 
            'hidden_w_init': nn.init.xavier_uniform_, 
            'output_w_inits': nn.init.xavier_uniform_, 
            'output_nonlinearity': nn.Identity()}
        zeta_learning_rate: float, 0.0001
            Learning rate for zeta.
        zeta_pos: bool
            Whether to enforce positivity constraint.
        solve_for_state_action_ratio: bool, True
            Whether to solve for state-action density ratio. 
        f_exponent: float
            Exponent p to use for f(x) = |x|^p / p.
        primal_form: bool, False
            Whether to use primal form of DualDICE, which optimizes for
            nu independent of zeta. This form is biased in stochastic environments.
            Defaults to False, which uses the saddle-point formulation of DualDICE.
        print_freq: int
            Frequency to print loss values
        verbose: bool
            Whether to output intermediate results
        """
        if not estimate_missing_prob:
            self.replay_buffer = SimpleReplayBuffer(trajs=self.masked_buffer, max_T=self.max_T - self.burn_in, 
                                                seed=self.seed)
        else:
            assert hasattr(
                self, 'propensity_pred'
            ), 'please call function self.estimate_missing_prob() first'
            self.replay_buffer = SimpleReplayBuffer(trajs=self.masked_buffer, max_T=self.max_T - self.burn_in, 
                                                prop_info=self.propensity_pred,
                                                seed=self.seed)

        self.verbose = verbose
        self.target_policy = target_policy
        self.initial_state_sampler = initial_state_sampler
        self._solve_for_state_action_ratio = solve_for_state_action_ratio
        if not solve_for_state_action_ratio:
            raise NotImplementedError
        self._categorical_action = True  # currently only handles discrete action space
        self._primal_form = primal_form
        self.ipw = ipw
        self.prob_lbound = prob_lbound

        self._nu_network = QNetwork(input_dim=self.state_dim,
                                    output_dim=self.num_actions,
                                    **nu_network_kwargs)
        self._nu_optimizer = torch.optim.Adam(
            params=self._nu_network.parameters(), lr=nu_learning_rate)

        if zeta_pos and 'output_nonlinearity' not in zeta_network_kwargs:
            zeta_network_kwargs['output_nonlinearity'] = Square() # Log1pexp()
        self._zeta_network = QNetwork(input_dim=self.state_dim,
                                    output_dim=self.num_actions,
                                    **zeta_network_kwargs)
        self._zeta_optimizer = torch.optim.Adam(
            params=self._zeta_network.parameters(), lr=zeta_learning_rate)

        if f_exponent <= 1:
            raise ValueError('Exponent for f must be greater than 1.')
        fstar_exponent = f_exponent / (f_exponent - 1)
        self._f_fn = lambda x: torch.abs(x)**f_exponent / f_exponent
        self._fstar_fn = lambda x: torch.abs(x
                                             )**fstar_exponent / fstar_exponent

        if scaler == "NormCdf":
            self.scaler = normcdf()
        elif scaler == "Identity":
            self.scaler = iden()
        elif scaler == "MinMax":
            self.scaler = MinMaxScaler(
                min_val=self.env.low, max_val=self.env.high
            ) if self.env is not None else MinMaxScaler()
        elif scaler == "Standard":
            self.scaler = StandardScaler(with_mean=True, with_std=True)
        else:
            # a path to a fitted scaler
            assert os.path.exists(scaler)
            with open(scaler, 'rb') as f:
                self.scaler = pickle.load(f)

        # fit the scaler
        state = self.replay_buffer.states
        next_state = self.replay_buffer.next_states
        if state.ndim == 1:
            state = state.reshape((-1, 1))
            next_state = next_state.reshape((-1, 1))
        state_concat = np.vstack([state, next_state])
        self.scaler.fit(state_concat)

        self._train(max_iter=max_iter,
                    batch_size=batch_size,
                    print_freq=print_freq)

    def _get_average_value(self, network, states, policy):
        """
        Parameters
        ----------
        network: nn.Module
        states: torch.FloatTensor
        policy: DiscretePolicy
        """
        action_weights = policy.get_action_prob(
            states)  # input should be on the original scale
        action_weights = torch.Tensor(action_weights).to(self.device)
        return torch.sum(network(states) * action_weights, dim=1, keepdim=True)

    def _train_loss(self, states, actions, next_states, initial_states, weights=None):
        """
        Parameters
        ----------
        states: torch.FloatTensor, dim=(n,state_dim)
        actions: torch.LongTensor, dim=(n,1)
        next_states: torch.FloatTensor, dim=(n,state_dim)
        initial_states: torch.FloatTensor, dim=(n,state_dim)
        weights: torch.FloatTensor, dim=(n,1)
        """
        if not self.ipw or weights is None:
            weights = torch.ones_like(actions).float()

        if self._solve_for_state_action_ratio:
            nu_values = self._nu_network(states).gather(dim=1, index=actions)
            zeta_values = self._zeta_network(states).gather(dim=1, index=actions)
        else:
            nu_values = self._nu_network(states)
            zeta_values = self._zeta_network(states)
        initial_nu_values = self._get_average_value(self._nu_network,
                                                    initial_states,
                                                    self.target_policy)
        next_nu_values = self._get_average_value(self._nu_network, next_states,
                                                 self.target_policy)

        bellman_residuals = nu_values - self.gamma * next_nu_values

        zeta_loss = self._fstar_fn(zeta_values) - \
            bellman_residuals.detach() * zeta_values
        zeta_loss = torch.mean(zeta_loss * weights)
        
        nu_loss = - (1 - self.gamma) * torch.mean(initial_nu_values)
        if self._primal_form:
            nu_loss += torch.mean(self._f_fn(bellman_residuals) * weights)
        else:
            nu_loss += torch.mean(bellman_residuals * zeta_values.detach() * weights)
        return nu_loss, zeta_loss

    def _train(self, max_iter: int, batch_size: int, print_freq: int = 50):
        self._nu_network.train()
        self._zeta_network.train()

        dropout_prob = self.replay_buffer.dropout_prob
        if not self.ipw:
            dropout_prob = np.zeros_like(self.replay_buffer.actions)
        inverse_wts = 1 / np.clip(
            a=1 - dropout_prob, a_min=self.prob_lbound, a_max=1).astype(float)
        print('MaxInverseWeight:', np.max(inverse_wts))
        print('MinInverseWeight:', np.min(inverse_wts))

        if self.ipw:
            self.total_T_ipw = self.n * (self.max_T - self.burn_in) # self.n * (self.max_T - self.burn_in - 1)
        else:
            self.total_T_ipw = self.replay_buffer.N

        running_nu_losses = []
        running_zeta_losses = []

        for i in range(max_iter):
            transitions = self.replay_buffer.sample(batch_size)
            
            # num_sample_trajs = max(32, batch_size//self.max_T)
            # transitions = self.replay_buffer.sample_trajs(num_trajs=num_sample_trajs)
            
            states, actions, rewards, next_states, dropout_prob = transitions[:5]
            initial_states = self.initial_state_sampler.sample(batch_size)
            
            states = self.scaler.transform(states)
            next_states = self.scaler.transform(next_states)
            initial_states = self.scaler.transform(initial_states)
            
            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)
            if dropout_prob.ndim == 1:
                dropout_prob = dropout_prob.reshape(-1, 1)

            if not self.ipw:
                dropout_prob = np.zeros_like(actions)
            inverse_wts = 1 / np.clip(a=1 - dropout_prob, a_min=self.prob_lbound, a_max=1).astype(float)

            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            initial_states = torch.FloatTensor(initial_states).to(self.device)
            inverse_wts = torch.FloatTensor(inverse_wts).to(self.device)
            
            inverse_wts_scale = self.replay_buffer.N / self.total_T_ipw
            # inverse_wts_scale = len(states) / (num_sample_trajs * (self.max_T - self.burn_in))
            inverse_wts *= inverse_wts_scale
            
            nu_loss, zeta_loss = self._train_loss(
                states=states, 
                actions=actions, 
                next_states=next_states,
                initial_states=initial_states, 
                weights=inverse_wts)      

            # optimize nu and zeta network
            self._nu_optimizer.zero_grad()
            nu_loss.backward()
            self._nu_optimizer.step()

            self._zeta_optimizer.zero_grad()
            zeta_loss.backward()
            self._zeta_optimizer.step()

            running_nu_losses.append(nu_loss.item())
            running_zeta_losses.append(zeta_loss.item())

            if i % print_freq == 0:
                moving_window = 100
                if i >= moving_window:
                    mean_nu_loss = np.mean(
                        running_nu_losses[(i - moving_window + 1):i + 1])
                    mean_zeta_loss = np.mean(
                        running_zeta_losses[(i - moving_window + 1):i + 1])
                else:
                    mean_nu_loss = np.mean(running_nu_losses[:i + 1])
                    mean_zeta_loss = np.mean(running_zeta_losses[:i + 1])
                print(
                    "omega(s,a) training {}/{}: nu_loss = {:.3f}, zeta_loss = {:.3f}"
                    .format(i, max_iter, mean_nu_loss, mean_zeta_loss))

    def omega(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        self._zeta_network.eval()
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        states = self.scaler.transform(states)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        with torch.no_grad():
            omega_values = self._zeta_network(states).gather(dim=1,
                                                             index=actions)

        return omega_values.squeeze().numpy()

    def get_value(self, verbose=True) -> float:
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        rewards = self.replay_buffer.rewards
        if self.ipw:
            dropout_prob = self.replay_buffer.dropout_prob  # (NT,)
        else:
            dropout_prob = np.zeros_like(actions, dtype=float)
        inverse_wts = 1 / np.clip(a=1 - dropout_prob,
                                  a_min=self.prob_lbound,
                                  a_max=1).astype(float)
        inverse_wts_scale = self.replay_buffer.N / self.total_T_ipw
        inverse_wts *= inverse_wts_scale

        est_omega = self.omega(states, actions)
        # est_omega = est_omega / np.mean(est_omega)  # normalize
        V_int_IS = 1 / (1 - self.gamma) * np.mean(est_omega * inverse_wts * rewards)
        # V_int_IS = 1 / (1 - self.gamma) * np.sum(est_omega * inverse_wts * rewards) / np.sum(inverse_wts)
        
        if verbose:
            print('omega: min = ', np.min(est_omega), 'max = ', np.max(est_omega))
            print("value_est = {:.3f}".format(V_int_IS))
        
        return V_int_IS


class NeuralDice(SimulationBase):
    """
    Pytorch reimplementation of https://github.com/google-research/dice_rl/blob/master/estimators/neural_dice.py
    """
    def __init__(self,
                 env,
                 n,
                 horizon,
                 eval_env=None,
                 discount=0.9,
                 device=torch.device('cpu'),
                 seed=0):

        super().__init__(env=env,
                         n=n,
                         horizon=horizon,
                         discount=discount,
                         eval_env=eval_env)

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.gamma = discount

        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def estimate_omega(
        self,
        target_policy,
        initial_state_sampler,
        nu_network,
        zeta_network,
        nu_learning_rate,
        zeta_learning_rate,
        zero_reward: bool = True,
        solve_for_state_action_ratio: bool = True,
        f_exponent: float = 1.5,
        primal_form: bool = False,
        num_samples: Optional[int] = None,
        primal_regularizer: float = 0.,
        dual_regularizer: float = 1.,
        norm_regularizer: float = 1.,
        nu_regularizer: float = 0.,
        zeta_regularizer: float = 0.,
        weight_by_gamma: bool = False,
        ipw: bool = False,
        estimate_missing_prob: bool = True,
        prob_lbound: float = 1e-3,
        scaler: str = "Identity",
        max_iter: int = 10000,
        batch_size: int = 1024,
        print_freq: int = 50,
        verbose: bool = False
    ):
        """
        Estimate the state-action visitation ratio.

        Args:
            target_policy (DiscretePolicy): The target policy to evaluate
            initial_state_sampler (InitialStateSampler): Sampler for initial states and actions.
            nu_network (nn.Module): The nu-value network.
            zeta_network (nn.Module): The zeta-value network.
            zero_reward (bool): Whether to include the reward in computing the residual.
            solve_for_state_action_ratio (bool): Whether to solve for state-action density ratio. Defaults to True.
            f_exponent (float): Exponent p to use for f(x) = |x|^p / p.
            primal_form (bool): Whether to use primal form of DualDICE, which optimizes for
                nu independent of zeta. This form is biased in stochastic environments.
                Defaults to False, which uses the saddle-point formulation of DualDICE.
            num_samples (int): Number of samples to take from policy to estimate average
                next nu value. If actions are discrete, this defaults to computing
                average explicitly. If actions are not discrete, this defaults to using
                a single sample.
            primal_regularizer (float): Weight of primal varibale regularizer.
            dual_regularizer (float): Weight of dual varibale regularizer.
            norm_regularizer (float): Weight of normalization constraint.
            nu_regularizer (float): Regularization coefficient on nu network.
            zeta_regularizer (float): Regularization coefficient on zeta network.
            weight_by_gamma (bool): Whether to weight nu and zeta losses by gamma ** step_num.
            verbose (bool): Whether to output intermediate results
        """
        if not estimate_missing_prob:
            self.replay_buffer = SimpleReplayBuffer(trajs=self.masked_buffer, max_T=self.max_T - self.burn_in, 
                                                seed=self.seed)
        else:
            assert hasattr(
                self, 'propensity_pred'
            ), 'please call function self.estimate_missing_prob() first'
            self.replay_buffer = SimpleReplayBuffer(trajs=self.masked_buffer, max_T=self.max_T - self.burn_in, 
                                                prop_info=self.propensity_pred,
                                                seed=self.seed)

        self.verbose = verbose
        self.target_policy = target_policy
        self.initial_state_sampler = initial_state_sampler
        self.ipw = ipw
        self.prob_lbound = prob_lbound
        
        self._nu_network = nu_network
        self._zeta_network = zeta_network
        self._zero_reward = zero_reward

        self._nu_optimizer = torch.optim.Adam(
            params=self._nu_network.parameters(), lr=nu_learning_rate)
        self._zeta_optimizer = torch.optim.Adam(
            params=self._zeta_network.parameters(), lr=zeta_learning_rate)

        self._nu_regularizer = nu_regularizer
        self._zeta_regularizer = zeta_regularizer
        self._weight_by_gamma = weight_by_gamma
        self._num_samples = num_samples
        self._solve_for_state_action_ratio = solve_for_state_action_ratio

        if not self._solve_for_state_action_ratio:
            # require acess to behavior policy
            raise NotImplementedError

        if f_exponent <= 1:
            raise ValueError('Exponent for f must be greater than 1.')
        fstar_exponent = f_exponent / (f_exponent - 1)
        self._f_fn = lambda x: torch.abs(x)**f_exponent / f_exponent
        self._fstar_fn = lambda x: torch.abs(x)**fstar_exponent / fstar_exponent

        self._categorical_action = True  # currently only handles discrete action space
        if not self._categorical_action and self._num_samples is None:
            self._num_samples = 1

        self._primal_form = primal_form
        self._primal_regularizer = primal_regularizer
        self._dual_regularizer = dual_regularizer
        self._norm_regularizer = norm_regularizer

        self._lam = Variable(torch.tensor(0.0), requires_grad=True)
        self._lam_optimizer = torch.optim.Adam(params=[self._lam], lr=nu_learning_rate)

        if scaler == "NormCdf":
            self.scaler = normcdf()
        elif scaler == "Identity":
            self.scaler = iden()
        elif scaler == "MinMax":
            self.scaler = MinMaxScaler(
                min_val=self.env.low, max_val=self.env.high
            ) if self.env is not None else MinMaxScaler()
        elif scaler == "Standard":
            self.scaler = StandardScaler(with_mean=True, with_std=True)
        else:
            # a path to a fitted scaler
            assert os.path.exists(scaler)
            with open(scaler, 'rb') as f:
                self.scaler = pickle.load(f)

        # fit the scaler
        state = self.replay_buffer.states
        next_state = self.replay_buffer.next_states
        if state.ndim == 1:
            state = state.reshape((-1, 1))
            next_state = next_state.reshape((-1, 1))
        state_concat = np.vstack([state, next_state])
        self.scaler.fit(state_concat)

        self._train(max_iter=max_iter,
                    batch_size=batch_size,
                    print_freq=print_freq)

    def _get_average_value(self, network, states, policy):
        """
        Args:
            network (nn.Module)
            states (torch.FloatTensor)
            policy (DiscretePolicy)

        Returns:
            torch.FloatTensor
        """
        action_weights = policy.get_action_prob(
            states)  # input should be on the original scale
        action_weights = torch.Tensor(action_weights).to(self.device)
        return torch.sum(network(states) * action_weights, dim=1, keepdim=True)

    def _orthogonal_regularization(self, network):
        """Orthogonal regularization.
        See equation (3) in https://arxiv.org/abs/1809.11096.
        
        Args:
            network (torch.nn.Module): the torch.nn model to apply regualization for.
        
        Returns:
            A regularization loss term.
        """
        orth_loss = torch.tensor(0.)
        with torch.enable_grad():
            for name, param in network.named_parameters():
                if 'bias' not in name:
                    param_flat = param.view(param.shape[0], -1)
                    prod = torch.mm(param_flat, param_flat.t())
                    orth_loss += torch.sum(torch.square(prod * (1 - torch.eye(param_flat.shape[0]))))
        return orth_loss

    def _train_loss(self, states, actions, rewards, next_states, initial_states, step_num=None, weights=None):
        """
        Args:
            states (torch.FloatTensor): dim=(n,state_dim)
            actions (torch.LongTensor): dim=(n,1)
            rewards (torch.LongTensor): dim=(n,1)
            next_states (torch.FloatTensor): dim=(n,state_dim)
            initial_states (torch.FloatTensor): dim=(n,state_dim)
            weights (torch.FloatTensor): dim=(n,1)
        """
        if not self.ipw or weights is None:
            weights = torch.ones_like(actions).float()

        if self._solve_for_state_action_ratio:
            nu_values = self._nu_network(states).gather(dim=1, index=actions)
            zeta_values = self._zeta_network(states).gather(dim=1, index=actions)
        else:
            nu_values = self._nu_network(states)
            zeta_values = self._zeta_network(states)           
        initial_nu_values = self._get_average_value(self._nu_network,
                                                    initial_states,
                                                    self.target_policy)
        next_nu_values = self._get_average_value(self._nu_network, next_states,
                                                 self.target_policy)
        discounts = self.gamma
        policy_ratio = 1.0
        bellman_residuals = discounts * policy_ratio * next_nu_values - nu_values - self._norm_regularizer * self._lam
        if not self._zero_reward:
            bellman_residuals += policy_ratio * rewards

        zeta_loss = -zeta_values * bellman_residuals
        nu_loss = (1 - self.gamma) * initial_nu_values
        lam_loss = self._norm_regularizer * self._lam
        if self._primal_form:
            nu_loss += self._fstar_fn(bellman_residuals) * weights
            lam_loss = lam_loss + self._fstar_fn(bellman_residuals) * weights
        else:
            nu_loss += zeta_values * bellman_residuals * weights
            lam_loss = lam_loss - (self._norm_regularizer * zeta_values * self._lam) * weights

        nu_loss += self._primal_regularizer * self._f_fn(nu_values)
        zeta_loss += self._dual_regularizer * self._f_fn(zeta_values)
        zeta_loss *= weights

        if self._weight_by_gamma:
            assert step_num is not None
            discount_weights = self.gamma ** step_num.float()
            discount_weights = discount_weights / (1e-6 + torch.mean(discount_weights))
            if discount_weights.ndim == 1:
                discount_weights = discount_weights.unsqueeze(dim=1)
            nu_loss *= discount_weights
            zeta_loss *= discount_weights

        nu_loss = torch.mean(nu_loss)
        zeta_loss = torch.mean(zeta_loss)
        lam_loss = torch.mean(lam_loss)

        return nu_loss, zeta_loss, lam_loss

    def _train(self, max_iter: int, batch_size: int, print_freq: int = 50):
        self._nu_network.train()
        self._zeta_network.train()

        dropout_prob = self.replay_buffer.dropout_prob
        if not self.ipw:
            dropout_prob = np.zeros_like(self.replay_buffer.actions)
        inverse_wts = 1 / np.clip(
            a=1 - dropout_prob, a_min=self.prob_lbound, a_max=1).astype(float)
        print('MaxInverseWeight:', np.max(inverse_wts))
        print('MinInverseWeight:', np.min(inverse_wts))
        if self.ipw:
            self.total_T_ipw = self.replay_buffer.num_trajs * self.replay_buffer.max_T
        else:
            self.total_T_ipw = len(self.replay_buffer.states)

        running_nu_losses = []
        running_zeta_losses = []
        running_lam_losses = []

        for i in range(max_iter):
            transitions = self.replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dropout_prob = transitions[:5]
            initial_states = self.initial_state_sampler.sample(batch_size)
            states = self.scaler.transform(states)
            next_states = self.scaler.transform(next_states)
            initial_states = self.scaler.transform(initial_states)
            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)
            if rewards.ndim == 1:
                rewards = rewards.reshape(-1, 1)
            if dropout_prob.ndim == 1:
                dropout_prob = dropout_prob.reshape(-1, 1)

            if not self.ipw:
                dropout_prob = np.zeros_like(actions)
            inverse_wts = 1 / np.clip(
                a=1 - dropout_prob, a_min=self.prob_lbound, a_max=1).astype(float)

            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            initial_states = torch.FloatTensor(initial_states).to(self.device)
            inverse_wts = torch.FloatTensor(inverse_wts).to(self.device)

            nu_loss, zeta_loss, lam_loss = self._train_loss(
                states=states, 
                actions=actions, 
                rewards=rewards, 
                next_states=next_states,
                initial_states=initial_states,
                weights=inverse_wts)

            nu_loss += self._nu_regularizer * self._orthogonal_regularization(
                self._nu_network)
            zeta_loss += self._zeta_regularizer * self._orthogonal_regularization(
                self._zeta_network)
            
            self._nu_optimizer.zero_grad()
            self._zeta_optimizer.zero_grad()
            self._lam_optimizer.zero_grad()

            nu_loss.backward(retain_graph=True, inputs=list(self._nu_network.parameters()))
            zeta_loss.backward(inputs=list(self._zeta_network.parameters()))
            lam_loss.backward(inputs=[self._lam])

            self._nu_optimizer.step()
            self._zeta_optimizer.step()
            self._lam_optimizer.step()

            running_nu_losses.append(nu_loss.item())
            running_zeta_losses.append(zeta_loss.item())
            running_lam_losses.append(lam_loss.item())

            if i % print_freq == 0:
                moving_window = 100
                if i >= moving_window:
                    mean_nu_loss = np.mean(
                        running_nu_losses[(i - moving_window + 1):i + 1])
                    mean_zeta_loss = np.mean(
                        running_zeta_losses[(i - moving_window + 1):i + 1])
                    mean_lam_loss = np.mean(
                        running_lam_losses[(i - moving_window + 1):i + 1])
                else:
                    mean_nu_loss = np.mean(running_nu_losses[:i + 1])
                    mean_zeta_loss = np.mean(running_zeta_losses[:i + 1])
                    mean_lam_loss = np.mean(running_lam_losses[:i + 1])
                print(
                    "omega(s,a) training {}/{}: nu_loss = {:.3f}, zeta_loss = {:.3f}, lambda_loss = {:.3f}"
                    .format(i, max_iter, mean_nu_loss, mean_zeta_loss, mean_lam_loss))


    def omega(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        self._zeta_network.eval()
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        states = self.scaler.transform(states)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        with torch.no_grad():
            omega_values = self._zeta_network(states).gather(dim=1,
                                                             index=actions)

        return omega_values.squeeze().numpy()

    def get_value(self, verbose=True) -> float:
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        rewards = self.replay_buffer.rewards
        if self.ipw:
            dropout_prob = self.replay_buffer.dropout_prob  # (NT,)
        else:
            dropout_prob = np.zeros_like(actions, dtype=float)

        if self.ipw:
            inverse_wts = 1 / np.clip(a=1 - dropout_prob,
                                    a_min=self.prob_lbound,
                                    a_max=1).astype(float)
        else:
            inverse_wts = np.ones_like(actions, dtype=float)

        est_omega = self.omega(states, actions)
        # V_int_IS = 1 / (1 - self.gamma) * np.mean(est_omega * rewards * inverse_wts)
        V_int_IS = 1 / (1 - self.gamma) * np.sum(est_omega * rewards * inverse_wts) / self.total_T_ipw
        
        if verbose:
            print('omega: min = {}, max = {}'.format(np.min(est_omega), np.max(est_omega)))
            print("value_est = {:.3f}".format(V_int_IS))

        return V_int_IS


