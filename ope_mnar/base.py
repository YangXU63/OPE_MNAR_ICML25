import numpy as np
import copy
import pandas as pd
import os
import gc
import joblib
import time
import collections
import pathlib
from operator import itemgetter
# RL environment
from gym.spaces import Tuple
from gym.vector.utils.spaces import batch_space
# ML model
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# visualization
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    iden,
    MinMaxScaler,
    MLPModule,
    GmmMNARClassifier,
    SemiparamMNARClassifier,
    ParamMNARClassifier,
    SimpleReplayBuffer
)


class SimulationBase(object):

    def __init__(self, env=None, n=500, horizon=None, discount=0.8, eval_env=None):
        """
        Args:
            env (gym.Env): dynamic environment
            n (int): the number of subjects (trajectories)
            horizon (int): the maximum length of trajectories
            discount (float): discount factor
            eval_env (gym.Env): dynamic environment to evaluate the policy, if not specified, use env
        """
        assert env is not None or eval_env is not None, "please provide env or eval_env"
        self.env = env
        self.vectorized_env = env.is_vector_env if env is not None else True
        if eval_env is None and env is not None:
            self.eval_env = copy.deepcopy(env)
        else:
            self.eval_env = eval_env
        self.n = n
        self.max_T = self.env.T if horizon is None else horizon # maximum horizon
        self.gamma = discount
        # self.behavior_policy = lambda S: self.env.action_space.sample() if self.env is not None else None  # uniform sample

        if self.env is not None:
            if self.vectorized_env:
                self.env.num_envs = n
                self.env.observation_space = batch_space(
                    self.env.single_observation_space, n=self.env.num_envs)
                self.env.action_space = Tuple(
                    (self.env.single_action_space, ) * self.env.num_envs)
                self.num_actions = self.env.single_action_space.n # the number of candidate discrete actions
                self.state_dim = self.env.single_observation_space.shape[0]
            else:
                self.num_actions = self.env.action_space.n # the number of candidate discrete actions
                self.state_dim = self.env.observation_space.shape[0]

        self.low = np.array([np.inf] * self.state_dim) # initial values, will be updated when generating trajectories
        self.high = np.array([-np.inf] * self.state_dim) # initial values, will be updated when generating trajectories
        self.masked_buffer = {} # masked data buffer, only observed data are included
        self.full_buffer = {}
        self.misc_buffer = {}  # hold any other information

    def sample_initial_states(self, size, from_data=False, seed=None):
        assert self.env is not None
        np.random.seed(seed)
        if not from_data:
            if self.vectorized_env:
                old_size = self.env.num_envs
                # reset size
                self.env.num_envs = size
                self.env.observation_space = batch_space(
                    self.env.single_observation_space, n=self.env.num_envs)
                self.env.action_space = Tuple(
                    (self.env.single_action_space, ) * self.env.num_envs)
                S_inits = self.env.observation_space.sample()
                # recover size
                self.env.num_envs = old_size
                self.env.observation_space = batch_space(
                    self.env.single_observation_space, n=self.env.num_envs)
                self.env.action_space = Tuple(
                    (self.env.single_action_space, ) * self.env.num_envs)
            else:
                S_inits = np.vstack([self.env.reset() for _ in range(size)])
            return S_inits
        else:
            assert hasattr(self, '_initial_obs'), "please generate data or import trajectories first"
            selected_id = np.random.choice(a=self.n, size=size)
            return self._initial_obs[selected_id]

    def sample_states(self, size, seed=None):
        assert hasattr(self, '_obs'), "please generate data or import trajectories first"
        np.random.seed(seed)
        selected_id = np.random.choice(a=len(self._obs), size=size)
        return self._obs[selected_id]

    def gen_single_traj(self,
                 policy=None,
                 S_init=None,
                 S_init_kwargs={},
                 A_init=None,
                 burn_in=0,
                 evaluation=False,
                 seed=None):
        """
        Generate a single trajectory. 

        Args:
            policy (callable): the policy to generate the trajectory
            S_init (np.ndarray): initial states
            S_init_kwargs (dict): other kwargs input to env.reset()
            A_init (np.ndarray): initial action, dimension should equal to the number of candidate actions
            burn_in (int): length of burn-in period
            evaluation (bool): if True, use eval_env to generate trajectory
            seed (int): random seed for env

        Returns:
            observations_traj (np.ndarray)
            action_traj (np.ndarray)
            reward_traj (np.ndarray)
            traj_len (int): length of the trajectory
            survival_prob_traj (np.ndarray)
            dropout_next_traj (np.ndarray)
            dropout_prob_traj (np.ndarray)
        """
        if policy is None:
            policy = self.behavior_policy if self.behavior_policy is not None else lambda S: self.env.action_space.sample(
            )
        if evaluation:
            env = self.eval_env
            max_T = env.T
        else:
            env = self.env
            max_T = self.max_T
        # initialize the state
        if seed is not None:
            env.seed(seed)
        if S_init is not None or not S_init_kwargs:
            S = env.reset(S_init, **S_init_kwargs)
        else:
            S = env.reset()
        S_traj = [S]
        A_traj = []
        reward_traj = []
        survival_prob_traj = []
        dropout_next_traj, dropout_prob_traj = [], []
        step = 1
        # while step <= env.T:
        while step <= max_T:
            self.low = np.minimum(self.low, np.array(S))
            self.high = np.maximum(self.high, np.array(S))
            if len(S_traj) == 1 and A_init is not None:
                A = A_init
            else:
                A = policy(S)
            if hasattr(A, '__iter__'):
                A = np.array(A).squeeze()
                assert len(
                    A
                ) == self.num_actions, f"len(A):{len(A)}, num_actions: {self.num_actions}, output of policy should match the number of actions"
                A = np.random.choice(range(self.num_actions), p=A).item()
            S_next, reward, done, env_infos = env.step(A)
            S_traj.append(S_next)
            A_traj.append(A)
            reward_traj.append(reward)
            survival_prob = env_infos.get('next_survival_prob', 1)
            survival_prob_traj.append(survival_prob)
            dropout_next = env_infos.get('dropout', None)
            dropout_next_traj.append(dropout_next)
            dropout_prob = env_infos.get('dropout_prob', None)
            dropout_prob_traj.append(dropout_prob)
            S = S_next
            if dropout_next:
                break
            step += 1
        traj_len = len(reward_traj)
        # convert to numpy.ndarray
        if traj_len == max_T and not dropout_next:
            S_traj = np.array(S_traj)
        else:
            S_traj = np.array(S_traj)[:traj_len]
        A_traj = np.array(A_traj)
        reward_traj = np.array(reward_traj)
        survival_prob_traj = np.array(survival_prob_traj).reshape(-1)
        dropout_next_traj = np.array(dropout_next_traj).reshape(-1)
        dropout_prob_traj = np.array(dropout_prob_traj).reshape(-1)
        
        if burn_in is None:
            return [
                S_traj, 
                A_traj, 
                reward_traj, 
                traj_len, 
                survival_prob_traj,
                dropout_next_traj, 
                dropout_prob_traj
            ]
        else:
            if traj_len > burn_in:
                return [
                    S_traj[burn_in:], 
                    A_traj[burn_in:], 
                    reward_traj[burn_in:],
                    traj_len - burn_in, 
                    survival_prob_traj[burn_in:],
                    dropout_next_traj[burn_in:], 
                    dropout_prob_traj[burn_in:]
                ]
            else:
                return []

    def gen_batch_trajs(self,
                        policy=None,
                        S_inits=None,
                        A_inits=None,
                        burn_in=0,
                        evaluation=False,
                        seed=None):
        """
        Generate a batch of trajectories. 

        Args: 
            policy (callable): the policy to generate trajectories
            S_inits (np.ndarray): initial states
            A_inits (np.ndarray): initial actions
            burn_in (int): the lenght of burn-in period
            evaluation (bool): if True, use eval_env
            seed (int): seed (int): random seed for env

        Returns:
            observations_traj (np.ndarray)
            action_traj (np.ndarray)
            reward_traj (np.ndarray)
            T (int)
            survival_prob_traj (np.ndarray)
            dropout_next_traj (np.ndarray)
            dropout_prob_traj (np.ndarray)

        TODO: organize the output into a dictionary
        """
        if A_inits is not None and len(A_inits.shape) == 1:
            A_inits = A_inits.reshape(-1, 1)
        if policy is None:
            policy = self.behavior_policy if self.behavior_policy is not None else lambda S: self.env.action_space.sample()
        if evaluation:
            env = self.eval_env
            max_T = env.T
        else:
            env = self.env
            max_T = self.max_T
        if seed is not None:
            env.seed(seed)
        # initialize the states
        if S_inits is not None:
            S = env.reset(S_inits)
        else:
            S = env.reset()
        survival_prob_traj = []
        dropout_next_traj, dropout_prob_traj = [], []
        step = 1
        # while step <= env.T:
        while step <= max_T:
            self.low = np.minimum(self.low, np.min(S, axis=0))
            self.high = np.maximum(self.high, np.max(S, axis=0))
            if step == 1 and A_inits is not None:
                A = A_inits  # (num_envs,1)
            else:
                A = policy(S)  # (num_envs,num_actions)
                if isinstance(A, tuple):
                    A = np.expand_dims(np.array(A), axis=1)
            if A.shape[1] > 1:
                assert A.shape[
                    1] == self.num_actions, "output of policy should match the number of actions"
                # sample an action based on the probability
                A = (A.cumsum(axis=1) > np.random.rand(
                    A.shape[0])[:, None]).argmax(axis=1)
            S, _, _, env_infos = env.step(actions=A)
            survival_prob = env_infos.get('next_survival_prob',
                                          np.ones(shape=(len(S), 1)))
            survival_prob_traj.append(survival_prob)
            dropout_next = env_infos.get(
                'dropout', np.zeros(shape=(len(S), 1), dtype=np.int8))
            dropout_next_traj.append(dropout_next)
            dropout_prob = env_infos.get('dropout_prob',
                                         np.zeros(shape=(len(S), 1)))
            dropout_prob_traj.append(dropout_prob)
            step += 1
        S_traj = env.states_history  # (num_envs, T, dim)
        A_traj = env.actions_history  # (num_envs, T)
        reward_traj = env.rewards_history  # (num_envs, T)
        survival_prob_traj = np.concatenate(survival_prob_traj,
                                            axis=1)  # (num_envs, T)
        dropout_next_traj = np.concatenate(dropout_next_traj,
                                           axis=1)  # (num_envs, T)
        dropout_prob_traj = np.concatenate(dropout_prob_traj,
                                           axis=1)  # (num_envs, T)

        S_traj_mask = env.states_history_mask  # (num_envs, T)
        # T denotes the actual length of the observed trajectories
        T = np.argmin(S_traj_mask, axis=1)
        T[T == 0] = self.max_T
        # output state, action, reward trajectory and T
        if burn_in is None:
            return [
                S_traj, A_traj, reward_traj, T, survival_prob_traj,
                dropout_next_traj, dropout_prob_traj, S_traj_mask
            ]
        else:
            if any(T > burn_in):
                observed_index = T > burn_in
                return [
                    S_traj[observed_index, burn_in:, :], 
                    A_traj[observed_index, burn_in:],
                    reward_traj[observed_index,burn_in:], 
                    T[observed_index] - burn_in,
                    survival_prob_traj[observed_index, burn_in:],
                    dropout_next_traj[observed_index, burn_in:],
                    dropout_prob_traj[observed_index, burn_in:], 
                    S_traj_mask[observed_index, burn_in:]
                ]
            else:
                return []

    def gen_masked_buffer(self,
                          policy=None,
                          n=None,
                          total_N=None,
                          S_inits=None,
                          S_inits_kwargs={},
                          burn_in=0,
                          seed=None):
        """
        Generate masked (observed) buffer data.

        Args:
            policy (callable): the policy to generate trajectories
            n (int): number of trajectories
            total_N (int): total number of observed tuples
            S_init (np.ndarray): initial states
            S_init_kwargs (dict): additional kwargs passed to env.reset()
            burn_in (int): length of burn-in period
            seed (int): random seed for env
        """
        self.burn_in = burn_in
        self.behavior_policy = policy
        if n is None:
            n = self.n
        if S_inits is not None:
            assert len(
                S_inits) == n, "the number of initial states should match n"
        if not self.vectorized_env:
            count = 0
            incomplete_cnt = 0
            if total_N is None:
                for i in range(n):
                    trajs = self.gen_single_traj(
                        policy=policy,
                        burn_in=burn_in,
                        S_init=S_inits[i] if S_inits is not None else None,
                        S_init_kwargs={
                            k: v[i]
                            for k, v in S_inits_kwargs.items()
                        } if S_inits_kwargs else {},
                        A_init=None,
                        evaluation=False)
                    if not trajs:
                        continue
                    self.masked_buffer[(i)] = trajs
                    if self.masked_buffer[(i)][3] < self.max_T:
                        incomplete_cnt += 1
                    count += self.masked_buffer[(i)][3]
            else:
                i = 0
                incomplete_cnt = 0
                while count < total_N:
                    trajs = self.gen_single_traj(
                        policy=policy,
                        burn_in=burn_in,
                        S_init=S_inits[i] if S_inits is not None else None,
                        S_init_kwargs={
                            k: v[i]
                            for k, v in S_inits_kwargs.items()
                        } if S_inits_kwargs else {},
                        A_init=None,
                        evaluation=False)
                    if not trajs:
                        continue
                    self.masked_buffer[(i)] = trajs
                    if self.masked_buffer[(i)][3] < self.max_T:
                        incomplete_cnt += 1
                    count += self.masked_buffer[(i)][3]
                    i += 1
                self.n = i
            self.total_N = count
        else:
            count = 0
            incomplete_cnt = 0
            if total_N is None:
                concat_trajs = self.gen_batch_trajs(policy=policy,
                                                         burn_in=burn_in,
                                                         S_inits=S_inits,
                                                         A_inits=None,
                                                         seed=seed,
                                                         evaluation=False)
                S_traj, A_traj, reward_traj, T, survival_prob_traj, dropout_next_traj, dropout_prob_traj, S_traj_mask = concat_trajs
                for i in range(len(S_traj)):
                    observed_state_index = np.where(S_traj_mask[i] == 1)[0]
                    observed_index = observed_state_index[
                        observed_state_index < self.max_T - burn_in].tolist()
                    observed_state_index = observed_state_index.tolist()
                    self.masked_buffer[(i)] = [
                        S_traj[i][observed_state_index],
                        # S_traj[i][observed_index],
                        A_traj[i][observed_index],
                        reward_traj[i][observed_index],
                        T[i],
                        survival_prob_traj[i][observed_index],
                        dropout_next_traj[i][observed_index],
                        dropout_prob_traj[i][observed_index]
                    ]
                    self.full_buffer[(i)] = [
                        S_traj[i], A_traj[i], reward_traj[i], T[i],
                        survival_prob_traj[i], dropout_next_traj[i],
                        dropout_prob_traj[i]
                    ]
                    if T[i] < self.max_T:
                        incomplete_cnt += 1
                    count += T[i]
            else:
                i = 0
                incomplete_cnt = 0
                while count < total_N:
                    concat_trajs = self.gen_batch_trajs(policy=policy,
                                                             burn_in=burn_in,
                                                             S_inits=None,
                                                             A_inits=None,
                                                             seed=seed,
                                                             evaluation=False)
                    S_traj, A_traj, reward_traj, T, survival_prob_traj, dropout_next_traj, dropout_prob_traj, S_traj_mask = concat_trajs
                    if count + np.sum(T) <= total_N:
                        sub_n = len(S_traj)
                    else:
                        sub_n = np.where(
                            T.cumsum() > total_N - count)[0].min() + 1
                    # for j in range(i, i + sub_n):
                    for j in range(sub_n):
                        # print(f'S_traj_mask.shape:{S_traj_mask.shape}')
                        observed_state_index = np.where(S_traj_mask[j] == 1)[0]
                        observed_index = observed_state_index[
                            observed_state_index < self.max_T -
                            burn_in].tolist()
                        observed_state_index = observed_state_index.tolist()
                        self.masked_buffer[(j + i)] = [
                            S_traj[j][observed_state_index],
                            # S_traj[j][observed_index],
                            A_traj[j][observed_index],
                            reward_traj[j][observed_index],
                            T[j],
                            survival_prob_traj[j][observed_index],
                            dropout_next_traj[j][observed_index],
                            dropout_prob_traj[j][observed_index]
                        ]
                        self.full_buffer[(j + i)] = [
                            S_traj[j], A_traj[j], reward_traj[j], T[j],
                            survival_prob_traj[j], dropout_next_traj[j],
                            dropout_prob_traj[j]
                        ]
                        if T[j] < self.max_T:
                            incomplete_cnt += 1
                        count += T[j]
                    i += sub_n
                self.n = i
            self.total_N = count
        self.dropout_rate = incomplete_cnt / self.n
        self.missing_rate = 1 - self.total_N / (self.n * self.max_T)
        initial_obs = []
        initial_actions = []
        all_obs = []
        for k in self.masked_buffer.keys():
            initial_obs.append(self.masked_buffer[k][0][0])
            initial_actions.append(self.masked_buffer[k][1][0])
            all_obs.append(self.masked_buffer[k][0])
        
        self._initial_obs = np.vstack(initial_obs)
        self._initial_actions = np.array(initial_actions)
        self._obs = np.vstack(all_obs)

    def export_buffer(self):
        """Convert unscaled trajectories in self.masked_buffer to a dataframe.
        """
        obs_dim = np.array(self.masked_buffer[next(iter(
            self.masked_buffer))][0]).shape[1]
        X_cols = [f'X{i}' for i in range(1, obs_dim + 1)]
        df = pd.DataFrame([])
        for k in self.masked_buffer.keys():
            nrows = len(self.masked_buffer[k][0])
            action_list = self.masked_buffer[k][1]
            if len(action_list) < nrows:
                nonterminal_state_list = self.masked_buffer[k][0][:-1]
            else:
                nonterminal_state_list = self.masked_buffer[k][0]
            tmp = pd.DataFrame(self.masked_buffer[k][0], columns=X_cols)
            tmp['id'] = np.repeat(k, nrows)
            if len(action_list) < nrows:
                tmp['action'] = np.append(action_list, [None])
                tmp['reward'] = np.append(self.masked_buffer[k][2], [None])
                tmp['surv_prob'] = np.append(self.masked_buffer[k][4], [None])
                tmp['dropout_prob'] = np.append(self.masked_buffer[k][6], [None])
            else:
                tmp['action'] = action_list
                tmp['reward'] = self.masked_buffer[k][2]
                tmp['surv_prob'] = self.masked_buffer[k][4]
                tmp['dropout_prob'] = self.masked_buffer[k][6]
            df = df.append(tmp)
        df = df.reset_index(drop=True)
        return df

    def import_buffer(self,
                      data=None,
                      data_filename=None,
                      static_state_cols=None,
                      dynamic_state_cols=None,
                      action_col='action',
                      reward_col='reward',
                      id_col='id',
                      dropout_col=None,
                      subsample_id=None,
                      reward_transform=None,
                      burn_in=0,
                      mnar_nextobs_var=None,
                      mnar_noninstrument_var=None,
                      mnar_instrument_var=None,
                      **kwargs):
        """Import unscaled trajectories from a dataframe to masked_buffer.

        Args:
            data (pd.DataFrame): dataframe to be imported
            data_filename (str): path to the table
            static_state_cols (list): column names of static features
            dynamic_state_cols (list): column names of dynamic features
            action_col (str): name of action column
            reward_col (str): name of reward column
            id_col (str): name of reward column
            dropout_col (str): name of the binary column that indicates dropout event
            subsample_id (list): ids of subsample to be imported
            reward_transform (callable): transformation funtion of reward
            burn_in (int): length of burn-in period
            mnar_nextobs_var (str): column name of the next observation for MNAR inference
            mnar_noninstrument_var (list): column names of noninstrument variable for MNAR inference
            mnar_instrument_var (str): column name of instrument variable for MNAR inference
            kwargs (dict): kwargs passed to pd.read_csv()        
        """
        assert data is not None or data_filename is not None, "please provide data or data_filename."
        if data is None:
            print(f'Import from {data_filename} to buffer')
            data = pd.read_csv(data_filename, **kwargs)

        state_cols = static_state_cols + dynamic_state_cols
        self.state_cols = state_cols
        self._df = data
        self.num_actions = data[action_col].nunique()
        self.state_dim = len(state_cols)
        self.burn_in = burn_in
        if subsample_id is None:
            id_list = data[id_col].unique().tolist()
        else:
            id_list = subsample_id
        num_trajs = len(id_list)
        self.masked_buffer = {} # observed data
        self.misc_buffer = {} # miscellaneous info
        if state_cols is None:
            state_cols = [c for c in data.columns if c.startswith('X')]
        incomplete_cnt = 0
        self.total_N = 0
        self._initial_obs = []
        self._initial_actions = []
        if 'custom_dropout_prob' not in data.columns and dropout_col in data.columns:
            print(f'use column {dropout_col} as dropout indicator')
        for i, id_ in enumerate(id_list):
            tmp = data.loc[data[id_col] == id_]
            S_traj = tmp[state_cols].values
            self._initial_obs.append(S_traj[0])
            A_traj = tmp[action_col].values
            self._initial_actions.append(A_traj[0])
            reward_traj = tmp[reward_col].values
            if reward_transform is not None:
                reward_traj = reward_transform(reward_traj)
            T = len(A_traj)
            if 'custom_dropout_prob' not in tmp.columns:
                if 'surv_prob' in tmp.columns:
                    survival_prob_traj = tmp['surv_prob'].values.astype(float)
                    dropout_prob_traj = 1 - survival_prob_traj[
                        1:] / survival_prob_traj[:-1]
                    dropout_prob_traj = np.append(np.array([0]),
                                                  dropout_prob_traj)
                else:
                    survival_prob_traj = None
                    dropout_prob_traj = None
                if dropout_col in tmp.columns:
                    dropout_next_traj = tmp[dropout_col].values
                elif len(reward_traj) < self.max_T:
                    dropout_next_traj = np.zeros_like(reward_traj)
                    dropout_next_traj[-1] = 1
                else:
                    dropout_next_traj = np.zeros_like(reward_traj)
            else:
                dropout_prob_traj = tmp['custom_dropout_prob'].values.astype(float)
                if dropout_prob_traj[-1] is None:
                    dropout_prob_traj[-1] = -1
                survival_prob_traj = np.append(
                    1, (1 - dropout_prob_traj).cumprod())[:-1]

                # dropout_index = np.where(
                #     np.random.uniform(
                #         low=0, high=1, size=dropout_prob_traj.shape) <
                #     dropout_prob_traj)[0]
                # dropout_next_traj = np.zeros_like(reward_traj)
                # if len(dropout_index) > 0:
                #     dropout_index = dropout_index.min()
                #     dropout_next_traj[dropout_index] = 1
                # else:
                #     dropout_index = self.max_T
                # if dropout_prob_traj[-1] < 0:
                #     dropout_prob_traj[-1] = np.nan

                assert dropout_col in tmp.columns
                dropout_next_traj = tmp[dropout_col].values
                dropout_index = np.where(dropout_next_traj == 1)[0]
                if len(dropout_index) > 0:
                    dropout_index = dropout_index.min()
                else:
                    dropout_index = self.max_T
                T = min(dropout_index + 1, self.max_T)
                S_traj = S_traj[:T]
                A_traj = A_traj[:T]
                reward_traj = reward_traj[:T]
                survival_prob_traj = survival_prob_traj[:T]
                dropout_next_traj = dropout_next_traj[:T]
                dropout_prob_traj = dropout_prob_traj[:T]

            self.total_N += T
            # if T < self.max_T:
            #     incomplete_cnt += 1
            incomplete_cnt += sum(dropout_next_traj)

            if T > burn_in:
                self.masked_buffer[(i)] = [
                    S_traj[burn_in:], 
                    A_traj[burn_in:], 
                    reward_traj[burn_in:],
                    T,
                    survival_prob_traj[burn_in:] if survival_prob_traj is not None else None,
                    dropout_next_traj[burn_in:], 
                    dropout_prob_traj[burn_in:] if dropout_prob_traj is not None else None
                ]
            if mnar_nextobs_var is not None or mnar_noninstrument_var is not None is not None or mnar_instrument_var is not None:
                self.misc_buffer[(i)] = {}
            if mnar_nextobs_var is not None:
                mnar_nextobs_arr = tmp[mnar_nextobs_var].values
                if len(mnar_nextobs_arr.shape) == 1:
                    mnar_nextobs_arr = mnar_nextobs_arr.reshape(-1, 1)
                self.misc_buffer[(i)]['mnar_nextobs_arr'] = np.vstack([
                    mnar_nextobs_arr[burn_in + 1:],
                    np.zeros(shape=(1, mnar_nextobs_arr.shape[1])) * np.nan
                ]) # shift one step
            if mnar_noninstrument_var is not None:
                mnar_noninstrument_arr = tmp[mnar_noninstrument_var].values
                self.misc_buffer[(
                    i
                )]['mnar_noninstrument_arr'] = mnar_noninstrument_arr[burn_in:]
            if mnar_instrument_var is not None:
                mnar_instrument_arr = tmp[mnar_instrument_var].values
                self.misc_buffer[(
                    i)]['mnar_instrument_arr'] = mnar_instrument_arr[burn_in:]

        self.n = len(self.masked_buffer)
        self.dropout_rate = incomplete_cnt / self.n
        self.missing_rate = 1 - self.total_N / (self.n * self.max_T)
        self._initial_obs = np.array(self._initial_obs)
        self._initial_actions = np.array(self._initial_actions)
        print('Import finished!')

    def import_holdout_buffer(self,
                              data=None,
                              data_filename=None,
                              static_state_cols=None,
                              dynamic_state_cols=None,
                              action_col='action',
                              reward_col='reward',
                              id_col='id',
                              dropout_col=None,
                              reward_transform=None,
                              burn_in=0,
                              **kwargs):
        """Import unscaled trajectories from a dataframe to self.holdout_buffer. 
        This buffer is used to estimate behavior policy or learn optimal policy.

        Args:
            data (pd.DataFrame): dataframe to be imported
            data_filename (str): path to the table
            static_state_cols (list): column names of static features
            dynamic_state_cols (list): column names of dynamic features
            action_col (str): name of action column
            reward_col (str): name of reward column
            id_col (str): name of reward column
            dropout_col (str): name of the binary column that indicates dropout event
            subsample_id (list): ids of subsample to be imported
            reward_transform (callable): transformation funtion of reward
            burn_in (int): length of burn-in period
            kwargs (dict): kwargs passed to pd.read_csv()  
        """
        if data is None:
            print(f'Import from {data_filename} to buffer')
            data = pd.read_csv(data_filename, **kwargs)

        state_cols = static_state_cols + dynamic_state_cols
        if hasattr(self, 'state_cols'):
            assert self.state_cols == state_cols
        if hasattr(self, 'num_actions'):
            assert self.num_actions == data[action_col].nunique()
        if hasattr(self, 'state_dim'):
            assert self.state_dim == len(state_cols)
        if hasattr(self, 'burn_in'):
            assert self.burn_in == burn_in

        id_list = data[id_col].unique().tolist()
        num_trajs = len(id_list)
        self.holdout_buffer = {}
        if state_cols is None:
            state_cols = [c for c in data.columns if c.startswith('X')]
        for i, id_ in enumerate(id_list):
            tmp = data.loc[data[id_col] == id_]
            S_traj = tmp[state_cols].values
            A_traj = tmp[action_col].values
            reward_traj = tmp[reward_col].values
            if reward_transform is not None:
                reward_traj = reward_transform(reward_traj)
            T = len(A_traj)
            if 'surv_prob' in tmp.columns:
                survival_prob_traj = tmp['surv_prob'].values
                dropout_prob_traj = 1 - survival_prob_traj[
                    1:] / survival_prob_traj[:-1]
                dropout_prob_traj = np.append(np.array([0]), dropout_prob_traj)
            else:
                survival_prob_traj = None
                dropout_prob_traj = None

            dropout_next_traj = np.zeros_like(reward_traj)
            if dropout_col in tmp.columns:
                dropout_next_traj = tmp[dropout_col].values
            elif len(dropout_next_traj) < self.max_T:
                dropout_next_traj[-1] = 1

            if T > burn_in:
                self.holdout_buffer[(i)] = [
                    S_traj[burn_in:], A_traj[burn_in:], reward_traj[burn_in:],
                    T, survival_prob_traj[burn_in:]
                    if survival_prob_traj is not None else None,
                    dropout_next_traj[burn_in:], dropout_prob_traj[burn_in:]
                    if dropout_prob_traj is not None else None
                ]
        print('Import finished!')

    def train_state_model(self,
                          model_type="linear",
                          train_ratio=0.8,
                          scale_obs=False,
                          export_dir=None,
                          pkl_filename="state_model.pkl",
                          seed=None,
                          **kwargs):
        """Train state transition model.

        Args:
            model_type (str): select from "linear", "mlp" and "rf"
            train_ratio (float): proportion of training set
            scale_obs (bool): if True, scale observation
            export_dir (str): directory to export model
            pkl_filename (str): filename for trained model
            seed (int): random seed for data spliting
            kwargs (dict): passed to model
        """
        # prepare data
        obs_list, next_obs_list, action_list = [], [], []
        self.fitted_state_model = {}
        if pkl_filename is not None:
            self.state_model_filename = os.path.join(export_dir, pkl_filename)
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
        for i in self.masked_buffer.keys():
            S_traj, A_traj = self.masked_buffer[i][:2]
            obs_list.append(S_traj[:-1])
            next_obs_list.append(S_traj[1:])
            T = S_traj[:-1].shape[0]
            action_list.append(A_traj[:T].reshape(-1, 1))
        obs = np.vstack(obs_list)  # (total_T, S_dim)
        next_obs = np.vstack(next_obs_list)  # (total_T, S_dim)
        if scale_obs:
            if not hasattr(self.scaler, 'data_min_') or np.min(
                    self.scaler.data_min_) == -np.inf or np.max(
                        self.scaler.data_max_) == np.inf:
                self.scaler.fit(np.vstack([obs, next_obs]))
            obs = self.scaler.transform(obs)
            next_obs = self.scaler.transform(next_obs)
            self.fitted_state_model['scaler'] = self.scaler
        else:
            self.fitted_state_model['scaler'] = iden()
        actions = np.vstack(action_list)  # (total_T, 1)
        X, y = np.hstack([obs, actions]), next_obs
        data = {}
        for a in range(self.num_actions):
            X_a, y_a = X[X[:, -1] == a, :-1], y[X[:, -1] == a]
            X_train, X_test, y_train, y_test = train_test_split(
                X_a, y_a, test_size=1 - train_ratio, random_state=seed)
            data[a] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }

        if model_type.lower() == "linear":
            for a in range(self.num_actions):
                linear_reg = LinearRegression(fit_intercept=True)
                linear_reg.fit(X=data[a]['X_train'],
                               y=data[a]['y_train'],
                               **kwargs)
                mse_error = mean_squared_error(
                    y_true=data[a]['y_test'],
                    y_pred=linear_reg.predict(X=data[a]['X_test']))
                print(f'MSE on test set with action {a}: {mse_error}')
                self.fitted_state_model[a] = linear_reg
        elif model_type.lower() == "mlp":
            for a in range(self.num_actions):
                # create a validation set out of test set
                X_val_a, X_test_a, y_val_a, y_test_a = train_test_split(
                    data[a]['X_test'],
                    data[a]['y_test'],
                    test_size=0.5,
                    random_state=seed)
                mlp_model = MLPModule(input_dim=self.state_dim,
                                      output_dim=self.state_dim,
                                      hidden_sizes=[64, 64],
                                      hidden_nonlinearity=nn.ReLU(),
                                      lr=2e-3,
                                      loss=F.mse_loss)
                early_stop_callback = EarlyStopping(monitor='valid_loss',
                                                    patience=3,
                                                    verbose=False,
                                                    mode='min')
                trainer = pl.Trainer(max_epochs=250,
                                     callbacks=[early_stop_callback],
                                     **kwargs)
                train_dataloader = DataLoader(
                    TensorDataset(torch.Tensor(data[a]['X_train']),
                                  torch.Tensor(data[a]['y_train'])),
                    batch_size=min(len(data[a]['X_train']), 128),
                    drop_last=True)
                val_dataloader = DataLoader(TensorDataset(
                    torch.Tensor(X_val_a), torch.Tensor(y_val_a)),
                                            batch_size=min(len(X_val_a), 32),
                                            drop_last=True)
                test_dataloader = DataLoader(TensorDataset(
                    torch.Tensor(X_test_a), torch.Tensor(y_test_a)),
                                             batch_size=len(X_test_a),
                                             drop_last=True)
                trainer.fit(model=mlp_model,
                            train_dataloader=train_dataloader,
                            val_dataloaders=val_dataloader)
                eval_output = trainer.test(model=mlp_model,
                                           test_dataloaders=test_dataloader,
                                           verbose=False)
                mse_error = eval_output[0]['test_loss']
                print(f'MSE on test set with action {a}: {mse_error}')
                self.fitted_state_model[a] = mlp_model
        elif model_type.lower() == "rf":
            for a in range(self.num_actions):
                rf_reg = RandomForestRegressor(random_state=seed, **kwargs)
                rf_reg.fit(X=data[a]['X_train'], y=data[a]['y_train'])
                mse_error = mean_squared_error(
                    y_true=data[a]['y_test'],
                    y_pred=rf_reg.predict(X=data[a]['X_test']))
                print(f'MSE on test set with action {a}: {mse_error}')
                self.fitted_state_model[a] = rf_reg
            del rf_reg
            _ = gc.collect()
        else:
            raise NotImplementedError

        joblib.dump(value=self.fitted_state_model,
                    filename=self.state_model_filename,
                    compress=3)

    def train_reward_model(self,
                           model_type="linear",
                           train_ratio=0.8,
                           scale_obs=False,
                           export_dir=None,
                           pkl_filename="reward_model.pkl",
                           seed=None,
                           **kwargs):
        """Train reward model.

        Args:
            model_type (str): select from "linear", "mlp" and "rf"
            train_ratio (float): proportion of training set
            scale_obs (bool): if True, scale observation
            export_dir (str): directory to export model
            pkl_filename (str): filename for trained model
            kwargs (dict): passed to model
        """
        # prepare data
        obs_list, next_obs_list, action_list, reward_list = [], [], [], []
        self.fitted_reward_model = {}
        if pkl_filename is not None:
            self.reward_model_filename = os.path.join(export_dir, pkl_filename)
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
        for i in self.masked_buffer.keys():
            S_traj, A_traj, R_traj = self.masked_buffer[i][:3]
            obs_list.append(S_traj[:-1])
            next_obs_list.append(S_traj[1:])
            T = S_traj[:-1].shape[0]
            action_list.append(A_traj[:T].reshape(-1, 1))
            reward_list.append(R_traj[:T].reshape(-1, 1))
        obs = np.vstack(obs_list)  # (total_T, S_dim)
        next_obs = np.vstack(next_obs_list)  # (total_T, S_dim)
        if scale_obs:
            if not hasattr(self.scaler, 'data_min_') or np.min(
                    self.scaler.data_min_) == -np.inf or np.max(
                        self.scaler.data_max_) == np.inf:
                self.scaler.fit(np.vstack([obs, next_obs]))
            obs = self.scaler.transform(obs)
            next_obs = self.scaler.transform(next_obs)
            self.fitted_reward_model['scaler'] = self.scaler
        else:
            self.fitted_reward_model['scaler'] = iden()
        actions = np.vstack(action_list).squeeze()  # (total_T, )
        rewards = np.vstack(reward_list).squeeze()  # (total_T, )
        X, y = np.hstack([obs, next_obs]), rewards
        data = {}
        for a in range(self.num_actions):
            X_a, y_a = X[actions == a], y[actions == a]
            X_train, X_test, y_train, y_test = train_test_split(
                X_a, y_a, test_size=1 - train_ratio, random_state=seed)
            data[a] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        if model_type == "linear":
            for a in range(self.num_actions):
                linear_reg = LinearRegression(fit_intercept=True)
                linear_reg.fit(X=data[a]['X_train'],
                               y=data[a]['y_train'],
                               **kwargs)
                mse_error = mean_squared_error(
                    y_true=data[a]['y_test'],
                    y_pred=linear_reg.predict(X=data[a]['X_test']))
                print(f'MSE on test set with action {a}: {mse_error}')
                self.fitted_reward_model[a] = linear_reg
        elif model_type.lower() == "mlp":
            for a in range(self.num_actions):
                # create a validation set out of test set
                X_val_a, X_test_a, y_val_a, y_test_a = train_test_split(
                    data[a]['X_test'],
                    data[a]['y_test'].reshape(-1, 1),
                    test_size=0.5,
                    random_state=seed)
                mlp_model = MLPModule(input_dim=self.state_dim * 2,
                                      output_dim=1,
                                      hidden_sizes=[64, 64],
                                      hidden_nonlinearity=nn.ReLU(),
                                      lr=1e-3,
                                      loss=F.mse_loss)
                early_stop_callback = EarlyStopping(monitor='valid_loss',
                                                    patience=3,
                                                    verbose=False,
                                                    mode='min')
                trainer = pl.Trainer(max_epochs=250,
                                     callbacks=[early_stop_callback],
                                     **kwargs)
                train_dataloader = DataLoader(
                    TensorDataset(torch.Tensor(data[a]['X_train']),
                                  torch.Tensor(data[a]['y_train'])),
                    batch_size=min(len(data[a]['X_train']), 128),
                    drop_last=True)
                val_dataloader = DataLoader(TensorDataset(
                    torch.Tensor(X_val_a), torch.Tensor(y_val_a)),
                                            batch_size=min(len(X_val_a), 32),
                                            drop_last=True)
                test_dataloader = DataLoader(TensorDataset(
                    torch.Tensor(X_test_a), torch.Tensor(y_test_a)),
                                             batch_size=len(X_test_a),
                                             drop_last=True)
                trainer.fit(model=mlp_model,
                            train_dataloader=train_dataloader,
                            val_dataloaders=val_dataloader)
                eval_output = trainer.test(model=mlp_model,
                                           test_dataloaders=test_dataloader,
                                           verbose=False)
                mse_error = eval_output[0]['test_loss']
                print(f'MSE on test set with action {a}: {mse_error}')
                self.fitted_reward_model[a] = mlp_model
        elif model_type == "rf":
            for a in range(self.num_actions):
                rf_reg = RandomForestRegressor(random_state=seed, **kwargs)
                rf_reg.fit(X=data[a]['X_train'], y=data[a]['y_train'])
                mse_error = mean_squared_error(
                    y_true=data[a]['y_test'],
                    y_pred=rf_reg.predict(X=data[a]['X_test']))
                print(f'MSE on test set with action {a}: {mse_error}')
                self.fitted_reward_model[a] = rf_reg
            del rf_reg  # rf model tends to take up a large amount of memory
            _ = gc.collect()
        else:
            raise NotImplementedError

        joblib.dump(value=self.fitted_reward_model,
                    filename=self.reward_model_filename,
                    compress=3)

    def train_dropout_model(self,
                            model_type="linear",
                            dropout_model = 'semiparam',
                            train_ratio=0.8,
                            scale_obs=False,
                            missing_mechanism='mar',
                            dropout_obs_count_thres=1,
                            subsample_index=None,
                            export_dir=None,
                            pkl_filename="dropout_model.pkl",
                            seed=None,
                            include_reward=False,
                            instrument_var_index=None,
                            mnar_y_transform=None,
                            psi_init=None,
                            parametric=False,
                            bandwidth_factor=1.5,
                            verbose=True,
                            **kwargs):
        """
        Train dropout model.

        Args:
            model_type (str): select from "linear", "mlp" and "rf"
            train_ratio (float): proportion of training set
            scale_obs (bool): if True, scale observation before fitting the model
            missing_mechanism (str): "mnar" or "mar"
            dropout_obs_count_thres (int): number of observations that is not subject to dropout
            subsample_index (list): ids of subsample to train the model
            export_dir (str): directory to export model
            pkl_filename (str): filename for trained model
            seed (int): random seed to split data
            include_reward (bool): if True, include reward as feature in dropuot model
            instrument_var_index (int): index of the instrument variable
            mnar_y_transform (callable): input next_obs and reward, output Y term for the mnar dropout model 
            psi_init (np.ndarray): initial value for psi in MNAR estimation
            parametric (bool): if True, use parametric model to estimate the missing propensity
            bandwidth_factor (float): the constant used in bandwidth calculation
            kwargs (dict): passed to model
        """
        assert missing_mechanism is not None, "please specify missing_mechanism as either 'mar' or 'mnar'."
        self.missing_mechanism = missing_mechanism
        self.dropout_model = dropout_model
        self.dropout_obs_count_thres = max(dropout_obs_count_thres - 1, 0)  # -1 because this is the index
        self._include_reward = include_reward
        self._mnar_y_transform = mnar_y_transform
        self._instrument_var_index = instrument_var_index
        self._scale_obs = scale_obs
        states_list, next_states_list, actions_list, rewards_list = [], [], [], []
        dropout_next_list, dropout_prob_list = [], []
        mnar_nextobs_list, mnar_noninstrument_list, mnar_instrument_list = [], [], []
        
        self.fitted_dropout_model = {}
        if export_dir is not None and pkl_filename is not None:
            self.dropout_model_filename = os.path.join(export_dir,
                                                       pkl_filename)
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.dropout_model_filename = None
        keys = self.masked_buffer.keys()
        if subsample_index is not None:
            keys = subsample_index
        if missing_mechanism == 'mar':
            for i in keys:
                S_traj, A_traj, R_traj, _, _, don_traj, dop_traj = self.masked_buffer[i][:7]
                traj_len = A_traj.shape[0]
                states = S_traj[self.dropout_obs_count_thres:traj_len]
                actions = A_traj[self.dropout_obs_count_thres:traj_len].reshape(-1, 1) # (traj_len,1)
                # use previous step of reward
                rewards = np.append(0, R_traj)[self.dropout_obs_count_thres:traj_len].reshape(
                                  -1, 1) # (traj_len,1)
                states_list.append(states)
                actions_list.append(actions)
                rewards_list.append(rewards)
                don = don_traj[self.dropout_obs_count_thres:traj_len].reshape(-1, 1)
                dropout_next_list.append(don)
                if dop_traj is not None:
                    dop = dop_traj[self.dropout_obs_count_thres:traj_len].reshape(-1, 1)
                    dropout_prob_list.append(dop)
        elif missing_mechanism == 'mnar':
            for i in keys:
                S_traj, A_traj, R_traj, _, _, don_traj, dop_traj = self.masked_buffer[i][:7]
                traj_len = A_traj.shape[0]
                if sum(don_traj) == 0: # no dropout
                    traj_len -= 1
                    if traj_len <= self.dropout_obs_count_thres:
                        continue # ?
                states = S_traj[self.dropout_obs_count_thres:traj_len]
                actions = A_traj[self.dropout_obs_count_thres:traj_len].reshape(-1, 1) # (traj_len,1)
                # use current reward
                rewards = R_traj[self.dropout_obs_count_thres:traj_len].reshape(-1, 1) # (traj_len,1)

                if len(S_traj) == traj_len:
                    next_states = np.vstack([
                            S_traj[self.dropout_obs_count_thres+1:traj_len+1],
                            [[np.nan] * self.state_dim]
                        ])
                else:
                    next_states = S_traj[self.dropout_obs_count_thres+1:traj_len+1]

                states_list.append(states)
                actions_list.append(actions)
                next_states_list.append(next_states)
                rewards_list.append(rewards)
                don = don_traj[self.dropout_obs_count_thres:traj_len].reshape(-1, 1)
                dropout_next_list.append(don)
                if dop_traj is not None:
                    dop = dop_traj[self.dropout_obs_count_thres:traj_len].reshape(-1, 1)
                    dropout_prob_list.append(dop)

                mnar_nextobs_traj = None
                mnar_noninstrument_traj = None
                mnar_instrument_traj = None                
                if self.misc_buffer:
                    mnar_nextobs_traj = self.misc_buffer[i].get(
                        'mnar_nextobs_arr', None)
                    mnar_noninstrument_traj = self.misc_buffer[i].get(
                        'mnar_noninstrument_arr', None)
                    mnar_instrument_traj = self.misc_buffer[i].get(
                        'mnar_instrument_arr', None)
                if mnar_nextobs_traj is not None:
                    mnar_nextobs_list.append(
                        mnar_nextobs_traj[self.dropout_obs_count_thres:traj_len])
                if mnar_noninstrument_traj is not None:
                    mnar_noninstrument_list.append(
                        mnar_noninstrument_traj[self.
                                                dropout_obs_count_thres:traj_len])
                if mnar_instrument_traj is not None:
                    mnar_instrument_list.append(
                        mnar_instrument_traj[self.dropout_obs_count_thres:traj_len])

        unscaled_states = states = np.vstack(states_list)  # (total_T, S_dim)
        unscaled_next_states = next_states = np.vstack(next_states_list) if next_states_list else None # (total_T, S_dim)
        actions = np.vstack(actions_list).squeeze()  # (total_T,)
        rewards = np.vstack(rewards_list).squeeze()  # (total_T,)
        dropout_next = np.vstack(dropout_next_list).squeeze()  # (total_T,)
        if dropout_prob_list:
            dropout_prob = np.vstack(dropout_prob_list).squeeze() # (total_T,)
        else:
            dropout_prob = None
        if scale_obs:
            if not hasattr(self.scaler, 'data_min_') or not hasattr(
                    self.scaler, 'data_max_') or np.min(
                        self.scaler.data_min_) == -np.inf or np.max(
                            self.scaler.data_max_) == np.inf:
                if next_states is not None:
                    self.scaler.fit(np.vstack([states, next_states]))
                else:
                    self.scaler.fit(states)
            states = self.scaler.transform(states)
            if next_states_list:
                next_states = self.scaler.transform(next_states)
            self.fitted_dropout_model['scaler'] = self.scaler
        else:
            self.fitted_dropout_model['scaler'] = iden()

        mnar_nextobs_arr = np.vstack(
            mnar_nextobs_list) if mnar_nextobs_list else None
        mnar_noninstrument_arr = np.vstack(
            mnar_noninstrument_list) if mnar_noninstrument_list else None
        mnar_instrument_arr = np.concatenate(
            mnar_instrument_list) if mnar_instrument_list else None

        if missing_mechanism == 'mar':
            if include_reward:
                X, y = np.hstack([states, rewards.reshape(-1, 1)]), dropout_next
            else:
                X, y = states, dropout_next
            data = {}
            states_train_list = [] 
            probT_train_list = []
            action_train_list = []
            for a in range(self.num_actions):
                X_a, y_a = X[actions == a], y[actions == a]
                # assert len(np.unique(y_a)) > 1 # ensure at least one positive label
                if dropout_prob is not None:
                    probT_a = dropout_prob[actions == a]
                    if train_ratio < 1:
                        X_train, X_test, y_train, y_test, states_train, states_test, probT_train, probT_test = train_test_split(
                            X_a,
                            y_a,
                            states[actions == a],
                            probT_a,
                            test_size=1 - train_ratio,
                            random_state=seed
                        )
                    else:
                        # use all data for training
                        X_train = X_test = X_a
                        y_train = y_test = y_a
                        states_train = states_test = states[actions == a]
                        probT_train = probT_test = probT_a
                    states_train_list.append(states_train)
                    probT_train_list.append(probT_train)
                else:
                    if train_ratio < 1:
                        X_train, X_test, y_train, y_test, states_train, states_test = train_test_split(
                            X_a, y_a, states[actions == a], test_size=1 - train_ratio, random_state=seed)
                    else:
                        X_train = X_test = X_a
                        y_train = y_test = y_a
                        states_train = states_test = states[actions == a]
                    probT_train, probT_test = None, None
                    states_train_list.append(states_train)
                data[a] = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'prob_train': probT_train,
                    'prob_test': probT_test
                }
                action_train_list.append([a] * len(X_train))
            states_train = np.vstack(states_train_list)
            action_train = np.concatenate(action_train_list)
            if probT_train_list:
                probT_train = np.concatenate(probT_train_list)
            if model_type.lower() == "linear":
                for a in range(self.num_actions):
                    if np.sum(data[a]['y_train']) == 0: # all 0's
                        self.fitted_dropout_model[a] = 0.
                        continue
                    elif np.sum(data[a]['y_train']) == len(data[a]['y_train']): # all 1's
                        self.fitted_dropout_model[a] = 1.
                        continue
                    logit_reg = LogisticRegression(fit_intercept=True,
                                                   **kwargs)
                    logit_reg.fit(X=data[a]['X_train'],
                                  y=data[a]['y_train'],
                                  **kwargs)
                    test_acc = accuracy_score(
                        y_true=data[a]['y_test'],
                        y_pred=logit_reg.predict(X=data[a]['X_test']))
                    if verbose:
                        print(
                            f'Accuracy on test set with action {a}: {test_acc}'
                        )
                    try:
                        test_auroc = roc_auc_score(
                            y_true=data[a]['y_test'],
                            y_score=logit_reg.predict_proba(
                                X=data[a]['X_test'])[:, 1])
                        if verbose:
                            print(
                                f'AUROC on test set with action {a}: {test_auroc}'
                            )
                    except:
                        pass
                    if data[a]['prob_test'] is not None:
                        nan_index = np.isnan(data[a]['prob_test'])
                        test_mse = mean_squared_error(
                            y_true=data[a]['prob_test'][~nan_index],
                            y_pred=logit_reg.predict_proba(
                                X=data[a]['X_test'])[:, 1][~nan_index])
                        if verbose:
                            print(
                                f'MSE on test set with action {a}: {test_mse}')
                    self.fitted_dropout_model[a] = logit_reg
            elif model_type.lower() == "rf":
                for a in range(self.num_actions):
                    if np.sum(data[a]['y_train']) == 0: # all 0's
                        self.fitted_dropout_model[a] = 0.
                        continue
                    elif np.sum(data[a]['y_train']) == len(data[a]['y_train']): # all 1's
                        self.fitted_dropout_model[a] = 1.
                        continue
                    rf_clf = RandomForestClassifier(random_state=seed,
                                                    **kwargs)
                    rf_clf.fit(X=data[a]['X_train'],
                               y=data[a]['y_train'],
                               **kwargs)
                    test_acc = accuracy_score(
                        y_true=data[a]['y_test'],
                        y_pred=rf_clf.predict(X=data[a]['X_test']))
                    if verbose:
                        print(
                            f'Accuracy on test set with action {a}: {test_acc}'
                        )
                    try:
                        test_auroc = roc_auc_score(
                            y_true=data[a]['y_test'],
                            y_score=rf_clf.predict_proba(
                                X=data[a]['X_test'])[:, 1])
                        if verbose:
                            print(
                                f'AUROC on test set with action {a}: {test_auroc}'
                            )
                    except:
                        pass
                    if data[a]['prob_test'] is not None:
                        nan_index = np.isnan(data[a]['prob_test'])
                        test_mse = mean_squared_error(
                            y_true=data[a]['prob_test'][~nan_index],
                            y_pred=rf_clf.predict_proba(
                                X=data[a]['X_test'])[:, 1][~nan_index])
                        if verbose:
                            print(
                                f'MSE on test set with action {a}: {test_mse}')
                    self.fitted_dropout_model[a] = rf_clf
                del rf_clf  # rf model tends to take up a large amount of memory
                _ = gc.collect()
            else:
                raise NotImplementedError
            self._dropout_prob_mse = test_mse if data[a][
                'prob_test'] is not None else None
            if self.dropout_model_filename is not None:
                joblib.dump(value=self.fitted_dropout_model,
                            filename=self.dropout_model_filename,
                            compress=3)
        elif missing_mechanism == 'mnar':
            # specify y_arr
            if include_reward:
                print('use the reward as outcome')
                y_arr = rewards # use reward as outcome
            elif mnar_nextobs_arr is not None:
                y_arr = mnar_nextobs_arr
            else:
                if mnar_y_transform is not None:
                    y_arr = mnar_y_transform(
                        np.hstack([unscaled_next_states, rewards.reshape(-1, 1)]))
                else:
                    y_arr = next_states # already scaled if set scale_obs=True
            if len(y_arr.shape) == 1 or y_arr.shape[1] == 1:
                y_dim = 1
            else:
                y_dim = y_arr.shape[1]
            y_arr = y_arr.reshape(-1, y_dim) # (n,y_dim)
            if self._scale_obs:
                self._y_arr_scaler = MinMaxScaler()
                self._y_arr_scaler.fit(y_arr)
                y_arr = self._y_arr_scaler.transform(y_arr)
            # specify u_arr
            if mnar_noninstrument_arr is None:
                if not isinstance(instrument_var_index,
                                  np.ndarray) and not hasattr(
                                      instrument_var_index, '__iter__'):
                    non_instrument_index = [
                        i for i in range(states.shape[1])
                        if i != instrument_var_index
                    ]
                else:
                    non_instrument_index = [
                        i for i in range(states.shape[1])
                        if i not in set(instrument_var_index)
                    ]
                u_dim = len(non_instrument_index)
                u_arr = states[:,
                            non_instrument_index].reshape(len(states),
                                                          u_dim)  # (n, u_dim)
            else:
                if len(mnar_noninstrument_arr.shape) == 1:
                    u_dim = 1
                else:
                    u_dim = mnar_noninstrument_arr.shape[1]
                u_arr = mnar_noninstrument_arr.reshape(-1, u_dim)
            if self._scale_obs:
                self._u_arr_scaler = MinMaxScaler()
                self._u_arr_scaler.fit(u_arr)
                u_arr = self._u_arr_scaler.transform(u_arr)
            
            if dropout_model == 'semiparam':
                # specify z_arr
                if mnar_instrument_arr is None:
                    assert instrument_var_index is not None, 'please provide the name of the instrument variable, otherwise there is identifibility issue.'
                    z_arr = states[:, instrument_var_index]  # (n,)
                    # discretize Z
                    L = y_dim + 3
                    z_min, z_max = z_arr.min(), z_arr.max()
                    self.instrument_disc_bins = np.quantile(
                        a=z_arr, q=np.linspace(0, 1, L + 1))  # divide by quantile
                    # make sure bins are non-overlap
                    bins_nonoverlap = []
                    for i in range(len(self.instrument_disc_bins)):
                        if not bins_nonoverlap or bins_nonoverlap[
                                -1] != self.instrument_disc_bins[i]:
                            bins_nonoverlap.append(self.instrument_disc_bins[i])
                    self.instrument_disc_bins = bins_nonoverlap
                    L = len(self.instrument_disc_bins) - 1
                    assert L >= y_dim + 1
                    # handle the boundary
                    self.instrument_disc_bins[0] -= 0.1
                    self.instrument_disc_bins[-1] += 0.1
                    z_arr = np.digitize(z_arr, bins=self.instrument_disc_bins)
                    z_arr_count = collections.Counter(z_arr)
                    # make sure every level of Z has samples in it
                    for i in range(1, L + 1):
                        assert z_arr_count[i] != 0
                else:
                    z_arr = mnar_instrument_arr.reshape(
                        -1)  # assume Z is already discretized
                    L = len(np.unique(z_arr))
                    # discretized Z start from 1 instead of 0
                    if min(z_arr) <= 0:
                        z_arr -= min(z_arr) - 1
                print(f'discretize IV into L={L} bins')
    
                if dropout_prob is not None:
                    probT_arr = dropout_prob.astype('float')
                    logitT_arr = np.log(1 / probT_arr - 1)
                else:
                    probT_arr, logitT_arr = None, None
                delta_arr = 1 - dropout_next
    
                if verbose:
                    print('count of dropout indicator:', sum(delta_arr == 0))
                    print(f'observed proportion: {np.mean(delta_arr)}')
    
                delta_train = None
                while delta_train is None or np.mean(delta_train) == 1:
                    if probT_arr is not None:
                        if train_ratio < 1:
                            u_train, u_test, z_train, z_test, states_train, states_test, y_train, y_test, delta_train, delta_test, probT_train, probT_test = train_test_split(
                                u_arr,
                                z_arr,
                                states,
                                y_arr,
                                delta_arr,
                                probT_arr,
                                test_size=1 - train_ratio,
                                random_state=seed)
                        else:
                            # use all data for training
                            u_train = u_test = u_arr
                            z_train = z_test = z_arr
                            states_train = states_test = states
                            y_train = y_test = y_arr
                            delta_train = delta_test = delta_arr
                            probT_train = probT_test = probT_arr
                    else:
                        if train_ratio < 1:
                            u_train, u_test, z_train, z_test, states_train, states_test, y_train, y_test, delta_train, delta_test = train_test_split(
                                u_arr,
                                z_arr,
                                states,
                                y_arr,
                                delta_arr,
                                test_size=1 - train_ratio,
                                random_state=seed)
                        else:
                            # use all data for training
                            u_train = u_test = u_arr
                            z_train = z_test = z_arr
                            states_train = states_test = states
                            y_train = y_test = y_arr
                            delta_train = delta_test = delta_arr
                        
                        probT_train, probT_test = None, None
                
                # train model
                if verbose:
                    print('Fit [MNAR] missing propensity model')
                    print('observed proportion (train): {}'.format(np.mean(delta_train)))
                if parametric:
                    mnar_clf = ParamMNARClassifier()
                    mnar_clf_kwargs = {}
                else:
                    mnar_clf = SemiparamMNARClassifier()
                    bounds = None
                    if psi_init is not None:
                        # bounds = ((psi_init - 1.5, psi_init + 1.5), ) # can set custom search range
                        bounds = tuple([(psi - 5, psi + 5) for psi in psi_init])
                    
                    mnar_clf_kwargs = {'psi_init': psi_init, 'bounds': bounds, 'bandwidth_factor': bandwidth_factor}
                
                fit_start = time.time()
                mnar_clf.fit(L=L,
                             z=z_train,
                             u=u_train,
                             y=y_train,
                             delta=delta_train,
                             seed=seed,
                             verbose=verbose,
                             **mnar_clf_kwargs)
                self.mnar_psi = mnar_clf.psi_hat
            else:
                if mnar_instrument_arr is None:
                    assert instrument_var_index is not None, 'please provide the name of the instrument variable, otherwise there is identifibility issue.'
                    z_arr = states[:, instrument_var_index]  # (n,) # instrument_var_index should be 1
                    
                else:
                    z_arr = mnar_instrument_arr.reshape(
                        -1)  
                # print(f'discretize IV into L={L} bins')
    
                if dropout_prob is not None: # yes it's not None, dim (n, )
                    probT_arr = dropout_prob.astype('float')
                    logitT_arr = np.log(1 / probT_arr - 1)
                else:
                    probT_arr, logitT_arr = None, None
                delta_arr = 1 - dropout_next
    
                if verbose:
                    print('count of dropout indicator:', sum(delta_arr == 0))
                    print(f'observed proportion: {np.mean(delta_arr)}')
    
                delta_train = None
                while delta_train is None or np.mean(delta_train) == 1:
                    if probT_arr is not None:
                        if train_ratio < 1:
                            u_train, u_test, z_train, z_test, states_train, states_test, y_train, y_test, delta_train, delta_test, probT_train, probT_test = train_test_split(
                                u_arr,
                                z_arr,
                                states,
                                y_arr,
                                delta_arr,
                                probT_arr,
                                test_size=1 - train_ratio,
                                random_state=seed)
                        else:
                            # use all data for training
                            u_train = u_test = u_arr
                            z_train = z_test = z_arr
                            states_train = states_test = states
                            y_train = y_test = y_arr
                            delta_train = delta_test = delta_arr
                            probT_train = probT_test = probT_arr
                    else:
                        if train_ratio < 1:
                            u_train, u_test, z_train, z_test, states_train, states_test, y_train, y_test, delta_train, delta_test = train_test_split(
                                u_arr,
                                z_arr,
                                states,
                                y_arr,
                                delta_arr,
                                test_size=1 - train_ratio,
                                random_state=seed)
                        else:
                            # use all data for training
                            u_train = u_test = u_arr
                            z_train = z_test = z_arr
                            states_train = states_test = states
                            y_train = y_test = y_arr
                            delta_train = delta_test = delta_arr
                        
                        probT_train, probT_test = None, None
                
                # train model
                if verbose:
                    print('Fit [MNAR] missing propensity model')
                    print('observed proportion (train): {}'.format(np.mean(delta_train)))
    
                mnar_clf = GmmMNARClassifier()
                # mnar_clf = NewtonMNARClassifier()
                mnar_clf_kwargs = {}
                fit_start = time.time()
                mnar_clf.fit(z=z_train,
                             u=u_train,
                             y=y_train,
                             delta=delta_train,
                             seed=seed,
                             psi_init=psi_init,
                             h=None,
                             verbose=verbose,
                             **mnar_clf_kwargs)
                
                self.mnar_psi = mnar_clf.psi_hat

                
            if verbose:
                print('Done! {} secs elapsed.'.format(time.time()-fit_start))
            self.fitted_dropout_model['model'] = mnar_clf
            # save the model
            if self.dropout_model_filename is not None:
                mnar_clf.save(self.dropout_model_filename)

            # the following part is only used for sanity check in simulation
            if True:
                prob_pred_test = 1 - \
                    mnar_clf.predict_proba(u=u_test, z=z_test, y=y_test)

                if verbose:
                    print(f'psi_hat: {mnar_clf.psi_hat}')
                    print(
                        f'max_prob_pred_test: {prob_pred_test[delta_test==1].max()}'
                    )
                    if probT_test is not None:
                        print(
                            f'max_probT_test: {probT_test[delta_test==1].max()}'
                        )
                try:
                    test_auroc = roc_auc_score(y_true=1 - delta_test,
                                               y_score=prob_pred_test)
                    if verbose:
                        print(f'AUROC on test set: {test_auroc}')
                except:
                    pass

                if probT_test is not None:
                    test_mse = mean_squared_error(
                        y_true=probT_test[delta_test == 1],
                        y_pred=prob_pred_test[delta_test == 1])
                    if verbose:
                        print(f'MSE on test set: {test_mse}')
                    self._dropout_prob_mse = test_mse
        else:
            raise NotImplementedError

        # some additional tracking
        if verbose and probT_train is not None:
            prob_pred_df = pd.DataFrame({
                'X1': states_train[:, 0],
                'X2': states_train[:, 1],
                'dropout_prob': probT_train
            })
            if missing_mechanism == 'mnar':
                prob_pred_train = 1 - \
                    mnar_clf.predict_proba(u=u_train, z=z_train, y=y_train)
            else:
                prob_pred_train = np.zeros_like(probT_train)
                for a in range(self.num_actions):
                    if isinstance(self.fitted_dropout_model[a], float):
                        prob_pred_train[action_train == a] = self.fitted_dropout_model[a]
                    else:
                        prob_pred_train[action_train == a] = self.fitted_dropout_model[a].predict_proba(X=data[a]['X_train'])[:, 1]
            prob_pred_df['dropout_prob_est'] = prob_pred_train
            if missing_mechanism == 'mnar':
                train_mse = mean_squared_error(
                    y_true=probT_train[delta_train == 1],
                    y_pred=prob_pred_train[delta_train == 1])
            else:
                nan_index = np.isnan(probT_train)
                train_mse = mean_squared_error(y_true=probT_train[~nan_index], y_pred=prob_pred_train[~nan_index])                
            logitT_train = np.log(1 / np.clip(probT_train, a_min=1e-8, a_max=1-1e-8) - 1)
            logit_pred_train = np.log(1 / np.clip(prob_pred_train, a_min=1e-8, a_max=1-1e-8) - 1)
            print(
                'true obs prob (0.0/0.25/0.5/0.75/1.0 quantile): {0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}'
                .format(np.nanmin(probT_train),
                        np.nanquantile(probT_train, 0.25),
                        np.nanquantile(probT_train, 0.5),
                        np.nanquantile(probT_train, 0.75),
                        np.nanmax(probT_train)))
            print(
                'true obs logit (0.0/0.25/0.5/0.75/1.0 quantile): {0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}'
                .format(np.nanmin(logitT_train),
                        np.nanquantile(logitT_train, 0.25),
                        np.nanquantile(logitT_train, 0.5),
                        np.nanquantile(logitT_train, 0.75),
                        np.nanmax(logitT_train)))
            print(
                'est obs prob (0.0/0.25/0.5/0.75/1.0 quantile): {0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}'
                .format(np.nanmin(prob_pred_train),
                        np.nanquantile(prob_pred_train, 0.25),
                        np.nanquantile(prob_pred_train, 0.5),
                        np.nanquantile(prob_pred_train, 0.75),
                        np.nanmax(prob_pred_train)))
            print(
                'est obs logit (0.0/0.25/0.5/0.75/1.0 quantile): {0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}'
                .format(np.nanmin(logit_pred_train),
                        np.nanquantile(logit_pred_train, 0.25),
                        np.nanquantile(logit_pred_train, 0.5),
                        np.nanquantile(logit_pred_train, 0.75),
                        np.nanmax(logit_pred_train)))

    def estimate_missing_prob(self,
                      missing_mechanism=None,
                      subsample_index=None):
        if missing_mechanism is not None:
            missing_mechanism = self.missing_mechanism
        assert hasattr(self, 'fitted_dropout_model'
                       ), 'please run function train_dropout_model() first'
        self.propensity_pred = {}
        keys = self.masked_buffer.keys()
        if subsample_index is not None:
            keys = subsample_index
        if missing_mechanism == 'mar':
            states_list = []
            actions_list = []
            rewards_list = []
            traj_idxs = []
            for i in keys:
                S_traj, A_traj, R_traj = self.masked_buffer[i][:3]
                traj_len = len(A_traj)
                states = S_traj[self.dropout_obs_count_thres:traj_len]
                actions = A_traj[self.dropout_obs_count_thres:traj_len] # (traj_len,)
                # use previous step of reward
                rewards = np.append(0, R_traj[:(
                    traj_len -
                    1)])[self.dropout_obs_count_thres:traj_len].reshape(-1, 1) # (traj_len,1)
                states_list.append(states)
                actions_list.append(actions)
                rewards_list.append(rewards)
                traj_idxs.extend([i] * len(actions))
            
            # concatenate together
            states = np.vstack(states_list)
            rewards = np.vstack(rewards_list)
            actions = np.concatenate(actions_list)
            traj_idxs = np.array(traj_idxs, dtype=int)
            states = self.fitted_dropout_model['scaler'].transform(states)
            if self._include_reward:
                X_mat = np.hstack([states, rewards])
            else:
                X_mat = states
            # predict dropout propensity
            dropout_prob_pred = np.zeros_like(actions, dtype=float)
            for a in range(self.num_actions):
                if any(actions == a) == False:
                    continue
                if isinstance(self.fitted_dropout_model[a], float):
                    dropout_prob_pred[actions == a] = self.fitted_dropout_model[a]
                else:
                    dropout_prob_pred[actions == a] = self.fitted_dropout_model[a].predict_proba(
                                X=X_mat[actions == a])[:, 1]
            
            # assign to each trajectory
            for i in keys:
                traj_dropout_prob_pred = np.append(np.array([0] * self.dropout_obs_count_thres),
                                 dropout_prob_pred[traj_idxs == i])
                traj_surv_prob_pred = (1 - traj_dropout_prob_pred).cumprod()
                self.propensity_pred[i] = [traj_dropout_prob_pred, traj_surv_prob_pred]
        elif missing_mechanism == 'mnar':
            states_list, next_states_list = [], []
            actions_list = []
            rewards_list = []
            traj_idxs = []
            y_list, u_list, z_list = [], [], []
            for i in keys:
                S_traj, A_traj, R_traj = self.masked_buffer[i][:3]
                traj_len = len(A_traj)
                states = S_traj[self.dropout_obs_count_thres:traj_len]
                actions = A_traj[self.dropout_obs_count_thres:traj_len] # (traj_len,)
                if len(S_traj) == traj_len:
                    next_states = np.vstack([
                        S_traj[self.dropout_obs_count_thres + 1:], 
                        [[np.nan] * self.state_dim]])
                else:
                    next_states = S_traj[self.dropout_obs_count_thres+1:traj_len+1]
                # use current reward
                rewards = R_traj[self.dropout_obs_count_thres:traj_len].reshape(-1, 1) # (traj_len,1)
                
                states_list.append(states)
                next_states_list.append(next_states)
                actions_list.append(actions)
                rewards_list.append(rewards)
                traj_idxs.extend([i] * len(actions))
                
                # dropout model related information
                mnar_nextobs_arr = None
                mnar_noninstrument_arr = None
                mnar_instrument_arr = None
                if self.misc_buffer:
                    mnar_nextobs_arr = self.misc_buffer[i].get(
                        'mnar_nextobs_arr', None)
                    mnar_noninstrument_arr = self.misc_buffer[i].get(
                        'mnar_noninstrument_arr', None)
                    mnar_instrument_arr = self.misc_buffer[i].get(
                        'mnar_instrument_arr', None)
                    if mnar_instrument_arr is not None and mnar_noninstrument_arr is not None \
                        and mnar_nextobs_arr is not None:
                        assert mnar_nextobs_arr.shape[0] == mnar_noninstrument_arr.shape[0] and \
                            mnar_nextobs_arr.shape[0] == mnar_instrument_arr.shape[0]
                
                # specify y_arr
                if self._include_reward:
                    y_arr = rewards # use reward as outcome
                elif mnar_nextobs_arr is None:
                    if self._mnar_y_transform is not None:
                        y_arr = self._mnar_y_transform(np.hstack([next_states, rewards]))
                    else:
                        y_arr = self.fitted_dropout_model['scaler'].transform(next_states)
                else:
                    y_arr = mnar_nextobs_arr[self.dropout_obs_count_thres:traj_len]
                if len(y_arr.shape) == 1 or y_arr.shape[1] == 1:
                    y_dim = 1
                else:
                    y_dim = y_arr.shape[1]
                y_arr = y_arr.reshape(-1, y_dim)
                if self._scale_obs:
                    assert hasattr(self, '_y_arr_scaler'), 'scaler for outcome not found'
                    y_arr = self._y_arr_scaler.transform(y_arr)
                y_list.append(y_arr)
                # specify u_arr
                if mnar_noninstrument_arr is None:
                    if not isinstance(self._instrument_var_index,
                                      np.ndarray) and not hasattr(
                                          self._instrument_var_index,
                                          '__iter__'):
                        non_instrument_index = [
                            i for i in range(states.shape[1])
                            if i != self._instrument_var_index
                        ]
                    else:
                        non_instrument_index = [
                            i for i in range(states.shape[1])
                            if i not in set(self._instrument_var_index)
                        ]
                    u_dim = len(non_instrument_index)
                    u_arr = states[:, non_instrument_index].reshape(len(states), u_dim)
                else:
                    u_arr = mnar_noninstrument_arr[
                        self.dropout_obs_count_thres:traj_len]
                    if len(u_arr.shape) == 1:
                        u_dim = 1
                    else:
                        u_dim = u_arr.shape[1]
                    u_arr = u_arr.reshape(-1, u_dim)
                if self._scale_obs:
                    assert hasattr(self, '_u_arr_scaler'), 'scaler for non-instrumental variables is not found'
                    u_arr = self._u_arr_scaler.transform(u_arr)
                u_list.append(u_arr)
                # specify z_arr
                if mnar_instrument_arr is None:
                    z_arr = states[:, self._instrument_var_index]
                    if self.dropout_model == 'semiparam':
                        z_arr = np.digitize(z_arr, bins=self.instrument_disc_bins)
                else:
                    z_arr = mnar_instrument_arr[
                        self.dropout_obs_count_thres:traj_len].reshape(-1)
                    # discretized Z start from 1 instead of 0
                    if self.dropout_model == 'semiparam':
                        if min(z_arr) <= 0:
                            z_arr -= min(z_arr) - 1
                z_list.append(z_arr)
        
            # concatenate together
            states = np.vstack(states_list)
            next_states = np.vstack(next_states_list)
            states = self.fitted_dropout_model['scaler'].transform(states)
            next_states = self.fitted_dropout_model['scaler'].transform(next_states)
            rewards = np.vstack(rewards_list)
            actions = np.concatenate(actions_list)
            traj_idxs = np.array(traj_idxs, dtype=int)
            y_arr = np.vstack(y_list)
            u_arr = np.vstack(u_list)
            z_arr = np.concatenate(z_list)
            
            # predict dropout propensity
            mnar_clf = self.fitted_dropout_model['model']
            dropout_prob_pred = 1 - \
                mnar_clf.predict_proba(u=u_arr, z=z_arr, y=y_arr)
            
            # assign to each trajectory
            for i in keys:
                traj_dropout_prob_pred = np.append(np.array([0] * self.dropout_obs_count_thres),
                                 dropout_prob_pred[traj_idxs == i])
                traj_surv_prob_pred = (1 - traj_dropout_prob_pred).cumprod()
                self.propensity_pred[i] = [traj_dropout_prob_pred, traj_surv_prob_pred]

    def estimate_behavior_policy(
            self,
            model_type="linear",
            train_ratio=0.8,
            export_dir=None,
            pkl_filename="behavior_policy_model.pkl",
            seed=None,
            verbose=True,
            use_holdout_data=True,
            **kwargs):
        """
        Estimate behavior policy.

        Args:
            model_type (str): select from "linear" and "rf"
            train_ratio (float): proportion of training set
            export_dir (str): directory to export model
            pkl_filename (str): filename for trained model
            seed (int): random seed to split data
            verbose (bool): if True, output intermediate outcome
            kwargs (dict): passed to model
        """
        action_list, f_list = [], []
        if export_dir is not None and pkl_filename is not None:
            self.behavior_model_filename = os.path.join(
                export_dir, pkl_filename)
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.behavior_model_filename = None
        if use_holdout_data:
            training_buffer = self.holdout_buffer
        else:
            training_buffer = self.masked_buffer
        keys = training_buffer.keys()
        for k in keys:
            S_traj, A_traj, reward_traj = training_buffer[k][:3]
            T = A_traj.shape[0]
            action_list.append(A_traj.reshape(-1, 1))
            X_mat = self.Q(states=S_traj, actions=A_traj, predictor=True)
            f_list.append(X_mat)
        actions = np.vstack(action_list).squeeze()
        X, y = np.vstack(f_list), actions
        data = {}
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=1 -
                                                            train_ratio,
                                                            random_state=seed)
        data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        if model_type.lower() == "linear":
            logit_reg = LogisticRegression(fit_intercept=True,
                                           max_iter=1000,
                                           **kwargs)
            logit_reg.fit(X=data['X_train'],
                          y=data['y_train'],
                          sample_weight=None)
            test_acc = accuracy_score(
                y_true=data['y_test'],
                y_pred=logit_reg.predict(X=data['X_test']))
            test_f1 = f1_score(y_true=data['y_test'],
                               y_pred=logit_reg.predict(X=data['X_test']),
                               average='macro')
            if verbose:
                print(f'Accuracy on test set: {test_acc}')
                print(f'F1 score on test set: {test_f1}')
            try:
                test_auroc = roc_auc_score(
                    y_true=data['y_test'],
                    y_score=logit_reg.predict_proba(X=data['X_test']),
                    multi_class='ovr')
                if verbose:
                    print(f'AUROC on test set: {test_auroc}')
            except:
                pass
            self.fitted_behavior_model = logit_reg
            del logit_reg
            _ = gc.collect()
        elif model_type.lower() == "rf":
            rf_clf = RandomForestClassifier(random_state=seed, **kwargs)
            rf_clf.fit(X=data['X_train'], y=data['y_train'])
            test_acc = accuracy_score(y_true=data['y_test'],
                                      y_pred=rf_clf.predict(X=data['X_test']))
            test_f1 = f1_score(y_true=data['y_test'],
                               y_pred=rf_clf.predict(X=data['X_test']),
                               average='macro')
            if verbose:
                print(f'Accuracy on test set: {test_acc}')
                print(f'F1 score on test set: {test_f1}')
            try:
                test_auroc = roc_auc_score(
                    y_true=data['y_test'],
                    y_score=rf_clf.predict_proba(X=data['X_test']),
                    multi_class='ovr')
                if verbose:
                    print(f'AUROC on test set: {test_auroc}')
            except:
                pass
            self.fitted_behavior_model = rf_clf
            del rf_clf  # rf model tends to take up a large amount of memory
            _ = gc.collect()
        else:
            raise NotImplementedError
        
        if self.behavior_model_filename is not None:
            joblib.dump(value=self.fitted_behavior_model,
                        filename=self.behavior_model_filename,
                        compress=3)

    def evaluate_pointwise_Q(
        self,
        policy,
        eval_size=20,
        eval_horizon=None,
        pointwise_eval_size=20,
        seed=None,
        S_inits=None,
        S_inits_kwargs={}
        ):
        """Evaluate pointwise Q and value

        Args:
            policy (callable): target policy to be evaluated
            eval_size (int): number of initial states to evaluate the policy
            eval_horizon (int): horizon of Monte Carlo approxiamation
            pointwise_eval_size (int): Monte Carlo size to estimate point-wise value
            seed (int): random seed passed to gen_batch_trajs()
            S_inits (np.ndarray): initial states to evaluate the policy
            S_inits_kwargs (dict): additional kwargs passed to env.reset()

        Returns:
            S_inits (np.ndarray): array of initial states to evaluate the policy
            true_value_avg (list): corresponding point-wise V(s)
            true_Q_dict (dict): corresponding point-wise Q(s,a), key is the action name
        """
        if S_inits is not None:
            if len(S_inits) == 1:
                S_inits = np.expand_dims(S_inits, axis=0)
            eval_size = len(S_inits)
        elif self.vectorized_env:
            S_inits = np.vstack([
                self.eval_env.single_observation_space.sample()
                for _ in range(eval_size)
            ])

        if self.vectorized_env:
            total_eval_size = pointwise_eval_size * eval_size
            if S_inits is not None:
                S_inits = np.repeat(S_inits,
                                    repeats=pointwise_eval_size,
                                    axis=0)
            old_num_envs = self.eval_env.num_envs
            old_horizon = self.eval_env.T
            # reset num_envs
            if eval_horizon is not None:
                self.eval_env.T = eval_horizon
            self.eval_env.num_envs = total_eval_size
            self.eval_env.observation_space = batch_space(
                self.eval_env.single_observation_space,
                n=self.eval_env.num_envs)
            self.eval_env.action_space = Tuple(
                (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

            print('eval V: generate trajectories')
            print('len(S_inits):', len(S_inits))
            trajectories = self.gen_batch_trajs(
                policy=policy,
                seed=seed,
                S_inits=S_inits if S_inits is not None else None,
                A_inits=None,
                burn_in=0,
                evaluation=True)
            rewards_history = self.eval_env.rewards_history
            print('eval V: calculate value')
            true_value = np.matmul(
                rewards_history,
                self.gamma**np.arange(start=0,
                                      stop=rewards_history.shape[1]).reshape(
                                          -1, 1))  # (total_eval_size, 1)
            true_value = true_value.reshape(eval_size, pointwise_eval_size)
            true_value_avg = true_value.mean(axis=1).tolist()
            self.eval_env.reset()
            del rewards_history
            _ = gc.collect()
            # get true Q for each action
            true_Q_dict = {}
            for a in range(self.eval_env.single_action_space.n):
                print('eval Q: generate trajectories')
                A_inits = np.repeat(a, repeats=len(S_inits))
                batch = self.gen_batch_trajs(
                    policy=policy,
                    seed=seed,
                    S_inits=S_inits if S_inits is not None else None,
                    A_inits=A_inits,
                    burn_in=0,
                    evaluation=True)
                rewards_history = self.eval_env.rewards_history * \
                    self.eval_env.states_history_mask[:,
                                                      :self.eval_env.rewards_history.shape[1]]
                print('eval Q: calculate value')
                true_Q = np.matmul(
                    rewards_history,
                    self.gamma**np.arange(
                        start=0, stop=rewards_history.shape[1]).reshape(-1, 1))
                true_Q = true_Q.reshape(eval_size, pointwise_eval_size)
                true_Q_avg = true_Q.mean(axis=1).tolist()
                true_Q_dict[a] = true_Q_avg

            # recover eval_env
            self.eval_env.T = old_horizon
            self.eval_env.num_envs = old_num_envs
            self.eval_env.observation_space = batch_space(
                self.eval_env.single_observation_space,
                n=self.eval_env.num_envs)
            self.eval_env.action_space = Tuple(
                (self.eval_env.single_action_space, ) * self.eval_env.num_envs)
            del trajectories, rewards_history
            _ = gc.collect()
            return S_inits[::pointwise_eval_size], true_value_avg, true_Q_dict
        else:
            old_horizon = self.eval_env.T
            if eval_horizon is not None:
                self.eval_env.T = eval_horizon
            print('eval_V: generate trajectories')
            self.eval_env.T = old_horizon
            true_value_list = []
            if S_inits is None and S_inits_kwargs:
                S_inits_sample = []
            for i in range(eval_size):
                rewards_histories_list = []
                for j in range(pointwise_eval_size):
                    traj = self.gen_single_traj(
                        policy=policy,
                        seed=None,
                        S_init=S_inits[i] if S_inits is not None else None,
                        S_init_kwargs={
                            k: v[i]
                            for k, v in S_inits_kwargs.items()
                        } if S_inits_kwargs else {},
                        A_init=None,
                        burn_in=0,
                        evaluation=True)
                    rewards_history = self.eval_env.rewards_history
                    rewards_histories_list.append(rewards_history)
                rewards_histories_arr = np.array(rewards_histories_list) # (pointwise_eval_size, eval_T)
                true_value = np.matmul(
                    rewards_histories_arr,
                    self.gamma**np.arange(
                        start=0, stop=rewards_histories_arr.shape[1]).reshape(
                            -1, 1)).reshape(-1)  # (pointwise_eval_size,)
                true_value_list.append(true_value)
                if S_inits is None and S_inits_kwargs:
                    S_inits_sample.append(traj[0][0])
            true_value_arr = np.array(true_value_list) # (eval_size,pointwise_eval_size)
            if S_inits is None and S_inits_kwargs:
                S_inits_sample = np.array(S_inits_sample)
                print(f'S_inits_sample: {S_inits_sample}')
            true_value_avg = true_value_arr.mean(
                axis=1).tolist()  # (eval_size,)
            self.eval_env.reset()
            del rewards_history, rewards_histories_list, rewards_histories_arr
            _ = gc.collect()
            # get true Q for each action
            true_Q_dict = {}
            for a in range(self.num_actions):
                print('eval_Q: generate trajectories')
                true_Q_list = []
                for i in range(eval_size):
                    rewards_histories_list = []
                    for j in range(pointwise_eval_size):
                        traj = self.gen_single_traj(
                            policy=policy,
                            seed=None,  # seed
                            S_init=S_inits[i] if S_inits is not None else None,
                            A_init=a,
                            burn_in=0,
                            evaluation=True)
                        rewards_history = self.eval_env.rewards_history
                        rewards_histories_list.append(rewards_history)
                    rewards_histories_arr = np.array(rewards_histories_list) # (pointwise_eval_size, eval_T)
                    true_Q = np.matmul(
                        rewards_histories_arr,
                        self.gamma**np.arange(
                            start=0,
                            stop=rewards_histories_arr.shape[1]).reshape(
                                -1))  # (pointwise_eval_size,)
                    true_Q_list.append(true_Q)
                true_Q_arr = np.array(true_Q_list) # (eval_size,pointwise_eval_size)
                true_Q_avg = true_Q_arr.mean(axis=1).tolist()  # (eval_size,)
                true_Q_dict[a] = true_Q_avg
            self.eval_env.T = old_horizon
            del rewards_history, rewards_histories_list, rewards_histories_arr
            _ = gc.collect()
            if S_inits is None and S_inits_kwargs:
                S_inits = S_inits_sample
            return S_inits, true_value_avg, true_Q_dict

    def evaluate_policy(self,
                        policy,
                        eval_size=20,
                        eval_horizon=None,
                        seed=None,
                        S_inits=None,
                        mask_unobserved=False,
                        repeats=1):
        """
        Evaluate given policy (true value) using Monte Carlo approximation

        Args:
            policy (callable): target policy to be evaluated
            eval_size (int): Monte Carlo size to estimate point-wise value
            seed (int): random seed passed to gen_single_traj() or gen_batch_trajs()
            S_inits (np.ndarray): initial states for policy evaluation
            mask_unobserved (bool): if True, mask unobserved states
            repeats (int): the number of trajectories generated for each state

        Returns:
            true_V (list): true value for each state
            action_percent: percentage of action=1 (only apply for binary action)
            est_V (list): estimated value for each state
            proportion of value within the bounds (float)
        """
        if not self.vectorized_env:
            true_V = []

            if S_inits is not None and np.shape(S_inits) == (self.env.dim, ):
                S_inits = np.tile(A=S_inits, reps=(eval_size, 1))
            if S_inits is not None:
                eval_size = len(S_inits)
            
            # reset eval_env
            old_horizon = self.eval_env.T
            if eval_horizon is not None:
                self.eval_env.T = eval_horizon
            # evaluation on n subjects
            for i in range(eval_size):
                trajectories = self.gen_single_traj(
                    policy=policy,
                    seed=seed,
                    S_init=S_inits[i] if S_inits is not None else None,
                    evaluation=True)
                S, A, reward, T = trajectories[:4]
                est_true_Value = sum(
                    np.array([self.gamma**j
                              for j in range(T)]) * np.array(reward))
                true_V.append(est_true_Value)
            
            # recover eval_env
            self.eval_env.T = old_horizon
            
            return np.mean(true_V) #, est_V
        else:
            if S_inits is not None and np.shape(S_inits) == (self.env.dim, ):
                S_inits = np.tile(A=S_inits, reps=(eval_size, 1))
            if S_inits is not None:
                eval_size = len(S_inits)
                ###################
                S_inits = np.repeat(a=S_inits, repeats=repeats, axis=0)
                ###################
            
            old_num_envs = self.eval_env.num_envs
            old_horizon = self.eval_env.T
            # reset eval_env
            if eval_horizon is not None:
                self.eval_env.T = eval_horizon
            self.eval_env.num_envs = int(eval_size * repeats)
            self.eval_env.observation_space = batch_space(
                self.eval_env.single_observation_space,
                n=self.eval_env.num_envs)
            self.eval_env.action_space = Tuple(
                (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

            trajectories = self.gen_batch_trajs(
                policy=policy,
                seed=seed,
                S_inits=S_inits if S_inits is not None else None,
                evaluation=True)
            S, A, reward, T = trajectories[:4]
            # set unobserved (mask=0) reward to 0
            if mask_unobserved:
                rewards_history = self.eval_env.rewards_history * self.eval_env.states_history_mask[:, :self.eval_env.rewards_history.shape[1]]
            else:
                rewards_history = self.eval_env.rewards_history
            # recover eval_env
            self.eval_env.T = old_horizon
            self.eval_env.num_envs = old_num_envs
            self.eval_env.observation_space = batch_space(
                self.eval_env.single_observation_space,
                n=self.eval_env.num_envs)
            self.eval_env.action_space = Tuple(
                (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

            true_value = np.matmul(
                rewards_history,
                self.gamma**np.arange(start=0,
                                      stop=rewards_history.shape[1]).reshape(
                                          -1, 1))
            ###################
            true_value = true_value.reshape(eval_size, repeats, order='C').mean(axis=1)
            ###################
            true_V = true_value.squeeze().tolist()

            return S_inits, true_V, np.mean(true_V)

    def get_state_values(self, target_policy, S_inits=None, MC_size=None):
        """Wrapper function of V
        
        Args:
            target_policy (callable): target policy to be evaluated.
            S_inits (np.ndarray): initial states for policy evaluation. If both S_inits and MC_size are not 
                specified, use initial observations from data.
            MC_size (int): sample size for Monte Carlo approximation.

        Returns:
            est_V (np.ndarray): value of target policy for each state
        """
        raise NotImplementedError

    def get_value(self, target_policy, S_inits=None, MC_size=None):
        """Wrapper function of V_int
        
        Args:
            target_policy (callable): target policy to be evaluated.
            S_inits (np.ndarray): initial states for policy evaluation. If both S_inits and MC_size are not 
                specified, use initial observations from data.
            MC_size (int): sample size for Monte Carlo approximation.

        Returns:
            est_V (float): integrated value of target policy
        """
        raise NotImplementedError

    def validate_visitation_ratio(self, grid_size=10, visualize=False, quantile=False, apply_weights=False, prefix=''):
        self.grid = []
        self.idx2states = collections.defaultdict(list)
        if not hasattr(self, 'replay_buffer'):
            self.replay_buffer = SimpleReplayBuffer(trajs=self.masked_buffer, seed=self.seed)
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        # predict omega
        omega_values = self.omega(states, actions)
        if not apply_weights:
            weights = np.ones_like(actions)
        else:
            weights = 1 - self.replay_buffer.dropout_prob
        omega_values *= weights
        
        discretized_states = np.zeros_like(states)
        for i in range(self.state_dim):
            if quantile:
                disc_bins = np.quantile(a=states[:,i], q=np.linspace(0, 1, grid_size + 1))
                disc_bins[0] -= 0.1
                disc_bins[-1] += 0.1
            else:
                disc_bins = np.linspace(start=self.low[i] - 0.1, stop=self.high[i] + 0.1, num=grid_size + 1)
            self.grid.append(disc_bins)
            discretized_states[:,i] = np.digitize(states[:,i], bins=disc_bins) - 1
        discretized_states = list(map(tuple, discretized_states.astype('int')))
        for ds, s, a, o in zip(discretized_states, states, actions, omega_values):
            # self.idx2states[ds].append(np.append(np.append(s, a), o))
            self.idx2states[ds].append(np.concatenate([s, [a], [o]]))

        # generate trajectories under the target policy
        self.idx2states_target = collections.defaultdict(list)

        init_states = self._initial_obs
        eval_size = len(init_states)
        if self.eval_env.is_vector_env:
            old_num_envs = self.eval_env.num_envs
            self.eval_env.num_envs = eval_size
            self.eval_env.observation_space = batch_space(
                self.eval_env.single_observation_space,
                n=self.eval_env.num_envs)
            self.eval_env.action_space = Tuple(
                (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

            trajectories = self.gen_batch_trajs(
                policy=self.target_policy.policy_func,
                seed=self.seed,
                S_inits=init_states,
                evaluation=True)
            states_target, actions_target, rewards_target = trajectories[:3]
            states_target = [s_traj[:len(a_traj)] for s_traj,a_traj in zip(states_target, actions_target)]
            time_idxs_target = np.tile(np.arange(self.eval_env.T), reps=len(states_target))
            assert len(states_target[0]) == len(actions_target[0]) == self.eval_env.T
            states_target = np.vstack(states_target)
            actions_target = actions_target.flatten()
            # print('empirical value:', np.mean(rewards_target) / (1 - self.gamma))
            
            # recover eval_env
            self.eval_env.num_envs = old_num_envs
            self.eval_env.observation_space = batch_space(
                self.eval_env.single_observation_space,
                n=self.eval_env.num_envs)
            self.eval_env.action_space = Tuple(
                (self.eval_env.single_action_space, ) * self.eval_env.num_envs)
        else:
            raise NotImplementedError

        discretized_states_target = np.zeros_like(states_target)
        for i in range(self.state_dim):
            disc_bins = self.grid[i]
            discretized_states_target[:,i] = np.digitize(states_target[:,i], bins=disc_bins) - 1
        discretized_states_target = list(map(tuple, discretized_states_target.astype('int')))
        for ds, s, a, t in zip(discretized_states_target, states_target, actions_target, time_idxs_target):
            self.idx2states_target[ds].append(np.concatenate([s, [a], [t]]))

        # only for  2D state, binary action
        freq_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
        visit_ratio_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
        for k, v in self.idx2states.items():
            v = np.array(v)
            freq_mat[0][k[0]][k[1]] = sum(v[:,self.state_dim] == 0) / len(states)
            freq_mat[1][k[0]][k[1]] = sum(v[:,self.state_dim] == 1) / len(states)
            visit_ratio_mat[0][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 0,self.state_dim+1]) if sum(v[:,self.state_dim] == 0) > 0 else 0
            visit_ratio_mat[1][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 1,self.state_dim+1]) if sum(v[:,self.state_dim] == 1) > 0 else 0

        freq_target_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
        visit_ratio_ref_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
        for k, v in self.idx2states_target.items():
            v = np.array(v)
            freq_target_mat[0][k[0]][k[1]] = (1 - self.gamma) * sum(self.gamma ** v[v[:,self.state_dim] == 0, self.state_dim+1]) / eval_size
            freq_target_mat[1][k[0]][k[1]] = (1 - self.gamma) * sum(self.gamma ** v[v[:,self.state_dim] == 1, self.state_dim+1]) / eval_size
            # freq_target_mat[1][k[0]][k[1]] = sum(v[:,self.state_dim] == 0) / len(states_target)
            # freq_target_mat[1][k[0]][k[1]] = sum(v[:,self.state_dim] == 1) / len(states_target)
            visit_ratio_ref_mat[0][k[0]][k[1]] = freq_target_mat[0][k[0]][k[1]] / max(freq_mat[0][k[0]][k[1]], 0.0001)
            visit_ratio_ref_mat[1][k[0]][k[1]] = freq_target_mat[1][k[0]][k[1]] / max(freq_mat[1][k[0]][k[1]], 0.0001)
    
        value_est = self.get_value(verbose=False)

        if visualize:

            fig, ax = plt.subplots(2, self.num_actions, figsize=(5*self.num_actions,8))
            for a in range(self.num_actions):
                sns.heatmap(
                    freq_mat[a], 
                    cmap="YlGnBu",
                    linewidth=1,
                    ax=ax[a,1]
                )
                ax[a,1].invert_yaxis()
                ax[a,1].set_title('discretized state visitation of pi_b (action={})'.format(a))
            for a in range(self.num_actions):
                sns.heatmap(
                    freq_target_mat[a], 
                    cmap="YlGnBu",
                    linewidth=1,
                    ax=ax[a,0]
                )
                ax[a,0].invert_yaxis()
                ax[a,0].set_title('discretized state visitation of pi (action={})'.format(a))
            plt.savefig('./output/{}visitation_heatplot.png'.format(prefix))

            fig, ax = plt.subplots(2, self.num_actions, figsize=(5*self.num_actions,8))
            for a in range(self.num_actions):
                sns.heatmap(
                    visit_ratio_mat[a], 
                    cmap="YlGnBu",
                    linewidth=1,
                    ax=ax[a,1]
                )
                ax[a,1].invert_yaxis()
                ax[a,1].set_title('est visitation ratio (action={})'.format(a))
            for a in range(self.num_actions):
                sns.heatmap(
                    visit_ratio_ref_mat[a], 
                    cmap="YlGnBu",
                    linewidth=1,
                    ax=ax[a,0]
                )
                ax[a,0].invert_yaxis()
                ax[a,0].set_title('empirical visitation ratio (action={})'.format(a))
            fig.suptitle('Value estimate: {:.3f}'.format(value_est))
            plt.savefig('./output/{}est_visitation_ratio_heatplot.png'.format(prefix))
            plt.close()

    def validate_Q(self, grid_size=10, visualize=False, quantile=False, slice_dim=(0,1), prefix=''):
        self.grid = []
        self.idx2states = collections.defaultdict(list)
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions

        repeats_per_state = 2 # 1
        states = np.repeat(a=states, repeats=repeats_per_state, axis=0)
        actions = np.repeat(a=actions, repeats=repeats_per_state, axis=0)

        Q_est = self.Q(states=states, actions=actions).squeeze()

        # generate trajectories under the target policy
        init_states = states # self._initial_obs
        init_actions = actions # self._init_actions
        old_num_envs = self.eval_env.num_envs
        eval_size = len(init_states)

        self.eval_env.num_envs = eval_size
        self.eval_env.observation_space = batch_space(
            self.eval_env.single_observation_space,
            n=self.eval_env.num_envs)
        self.eval_env.action_space = Tuple(
            (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

        trajectories = self.gen_batch_trajs(
            policy=self.target_policy.policy_func,
            seed=self.seed,
            S_inits=init_states,
            A_inits=init_actions,
            burn_in=0,
            evaluation=True)
        rewards_history = self.eval_env.rewards_history * \
                    self.eval_env.states_history_mask[:,
                                                      :self.eval_env.rewards_history.shape[1]]
        Q_ref = np.matmul(
                    rewards_history,
                    self.gamma**np.arange(
                        start=0, stop=rewards_history.shape[1]).reshape(-1, 1)).squeeze()
        # recover eval_env
        self.eval_env.num_envs = old_num_envs
        self.eval_env.observation_space = batch_space(
            self.eval_env.single_observation_space,
            n=self.eval_env.num_envs)
        self.eval_env.action_space = Tuple(
            (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

        discretized_states = np.zeros_like(states)
        for i in range(self.state_dim):
            if quantile:
                disc_bins = np.quantile(a=states[:,i], q=np.linspace(0, 1, grid_size + 1))
                disc_bins[0] -= 0.1
                disc_bins[-1] += 0.1
            else:
                disc_bins = np.linspace(start=self.low[i] - 0.1, stop=self.high[i] + 0.1, num=grid_size + 1)
            self.grid.append(disc_bins)
            discretized_states[:,i] = np.digitize(states[:,i], bins=disc_bins) - 1
        discretized_states = list(map(tuple, discretized_states.astype('int')))
        for ds, s, a, q, qr in zip(discretized_states, states, actions, Q_est, Q_ref):
            # only use the first 2 dimensions of discretized_states as key
            # self.idx2states[ds[:2]].append(np.concatenate([s, [a], [q], [qr]]))
            self.idx2states[itemgetter(*slice_dim)(ds)].append(np.concatenate([s, [a], [q], [qr]]))

        # only for binary action
        Q_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
        Q_ref_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
        for k, v in self.idx2states.items():
            v = np.array(v)
            if any(v[:,self.state_dim] == 0):
                Q_mat[0][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 0,self.state_dim+1])
                Q_ref_mat[0][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 0,self.state_dim+2])
            if any(v[:,self.state_dim] == 1):
                Q_mat[1][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 1,self.state_dim+1])
                Q_ref_mat[1][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 1,self.state_dim+2])

        if visualize:

            fig, ax = plt.subplots(2, self.num_actions, figsize=(5*self.num_actions,8))
            for a in range(self.num_actions):
                sns.heatmap(
                    Q_mat[a], 
                    cmap="YlGnBu",
                    linewidth=1,
                    ax=ax[a,1]
                )
                ax[a,1].invert_yaxis()
                ax[a,1].set_title('estimated Q (action={})'.format(a))
            for a in range(self.num_actions):
                sns.heatmap(
                    Q_ref_mat[a], 
                    cmap="YlGnBu",
                    linewidth=1,
                    ax=ax[a,0]
                )
                ax[a,0].invert_yaxis()
                ax[a,0].set_title('empirical Q (action={})'.format(a))
            plt.savefig('./output/{}Qfunc_heatplot.png'.format(prefix))