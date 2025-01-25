'''
Examine the point-estimator of value estimator.
'''

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import argparse
import pathlib
import time
import gc
import pickle
import gym
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

torch.manual_seed(0)  # for better reproducibility

try:
    from ope_mnar.utils import InitialStateSampler, DiscretePolicy, MinMaxScaler
    from custom_env.linear2d import Linear2dEnv, Linear2dVectorEnv
    from custom_env.cartpole import CartPoleEnv, CartPoleVectorEnv, DefaultCartPoleEnv
    from ope_mnar.importance_sampling import MWL, DualDice, NeuralDice
    from ope_mnar.direct_method import FQE, LSTDQ, MQL
    from ope_mnar.doubly_robust import DRL
    from ope_mnar.density import Square
    from ope_mnar.base import SimulationBase
    from batch_rl.dqn import QNetwork
except:
    import sys
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar/ope_mnar'))
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar'))
    from ope_mnar.utils import InitialStateSampler, DiscretePolicy, MinMaxScaler
    from custom_env.linear2d import Linear2dEnv, Linear2dVectorEnv
    from custom_env.cartpole import CartPoleEnv, CartPoleVectorEnv, DefaultCartPoleEnv
    from ope_mnar.importance_sampling import MWL, DualDice, NeuralDice
    from ope_mnar.direct_method import FQE, LSTDQ, MQL
    from ope_mnar.doubly_robust import DRL
    from ope_mnar.density import Square
    from ope_mnar.base import SimulationBase
    from batch_rl.dqn import QNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--env',
                    type=str,
                    default='cartpole',
                    choices=['linear2d', 'cartpole'])
parser.add_argument(
    '--method',
    type=str,
    default='lstdq',
    choices=['mwl', 'fqe', 'lstdq', 'dualdice', 'neuraldice', 'drl'])
parser.add_argument('--max_episode_length', type=int, default=25)  # {10, 25}
parser.add_argument('--num_trajs', type=int, default=500)  # {100, 500}
parser.add_argument('--discount', type=float, default=0.9)  # {0.8, 0.9}
parser.add_argument('--policy_id',
                    type=str,
                    default='0',
                    choices=['0', '1', '2', '3'])
parser.add_argument('--burn_in', type=int, default=0)
parser.add_argument('--mc_size', type=int, default=25)  # 250
parser.add_argument('--eval_policy_mc_size', type=int,
                    default=10000)  # {10000, 50000}
parser.add_argument('--eval_horizon', type=int, default=250)
parser.add_argument('--dropout_scheme',
                    type=str,
                    default='mnar.v0',
                    choices=[
                        '0', 'mar.v0', 'mar.v3', 'mnar.v0', 'mnar.v1',
                        'mnar.v2', 'mnar.v3'
                    ])
parser.add_argument('--dropout_rate', type=float, default=0.9)
parser.add_argument(
    '--dropout_obs_count_thres',
    type=int,
    default=2,
    help='the number of observations that is not subject to dropout')
parser.add_argument('--Q_func_class',
                    type=str,
                    default='spline',
                    choices=['nn', 'rf', 'spline', 'linear'])
parser.add_argument('--omega_func_class',
                    type=str,
                    default='spline',
                    choices=['nn', 'spline', 'expo_linear'])
parser.add_argument('--ipw',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=False) # False, True
parser.add_argument('--estimate_missing_prob',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=False)
parser.add_argument('--parametric_missing_prob',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=False)  # {False, True}
parser.add_argument('--weight_curr_step',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--vectorize_env',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
args = parser.parse_args()

if __name__ == '__main__':
    log_dir = os.path.expanduser('~/Projects/ope_mnar/output')

    env_class = args.env.lower()
    ope_method = args.method.lower()
    T = args.max_episode_length
    n = args.num_trajs
    policy_id = args.policy_id
    total_N = None
    CI_alphas = [0.05] # [0.4, 0.3, 0.2, 0.1, 0.05], should be a list
    if env_class == 'cartpole':
        # override
        T = 100 # 200
        n = 50 # 100
    gamma = args.discount
    mc_size = args.mc_size
    dropout_scheme = args.dropout_scheme
    dropout_rate = args.dropout_rate
    dropout_obs_count_thres = args.dropout_obs_count_thres
    burn_in = args.burn_in
    vectorize_env = args.vectorize_env
    ipw = args.ipw
    weight_curr_step = args.weight_curr_step  # if False, use survival probability
    estimate_missing_prob = args.estimate_missing_prob
    parametric_missing_prob = args.parametric_missing_prob
    eval_policy_mc_size = args.eval_policy_mc_size
    eval_horizon = args.eval_horizon
    eval_seed = 123
    prob_lbound = 1e-2
    vis_quantile = True # True  # only for visualization purpose

    # (optional) sensitivity analysis
    misspec_IV = False  # {True, False}
    misspec_MOD = False  # {True, False}

    # model-related configuration
    omega_func_class = args.omega_func_class  # marginalized density ratio
    Q_func_class = args.Q_func_class
    default_scaler = 'MinMax'  # 'NormCdf', 'MinMax'
    if ope_method == 'lstdq':
        Q_func_class = 'spline'  # override
    if omega_func_class in ['spline', 'expo_linear'
                            ] or Q_func_class == 'spline':
        adaptive_dof = False
        # spline related configuration
        spline_degree = 3
        product_tensor = True  # True
        if env_class == 'linear2d':
            basis_scale_factor = 1  # 100
            ridge_factor = 1e-5  # 1e-5 # 1e-3
            if adaptive_dof:
                dof = max(spline_degree + 1, int(
                    ((n * T)**(3 / 7))**(1 / 2)))  # degree of freedom
            else:
                dof = 6  #7
            # knots = np.linspace(start=-spline_degree / (dof - spline_degree),
            #                     stop=1 + spline_degree / (dof - spline_degree),
            #                     num=dof + spline_degree + 1)  # handle the boundary
            knots = 'quantile'  # place knots at equally spaced sample quantiles
        elif env_class == 'cartpole':
            spline_degree = 3
            product_tensor = False
            # spline_degree = 2 # 2
            # product_tensor = True
            basis_scale_factor = 1 # 10
            ridge_factor = 1e-5 # 1e-3
            if adaptive_dof:
                dof = max(spline_degree + 1, int(
                    ((n * T)**(3 / 7))**(1 / 4)))  # degree of freedom
            else:
                dof = spline_degree + 1 if product_tensor else 6 # 4
            knots = 'quantile' # 'quantile' 'equivdist'


    # dropout model configuration
    dropout_model_type = 'linear'
    instrument_var_index = None
    mnar_y_transform = None
    psi_true = None  # 1.5, None
    initialize_with_psiT = False
    bandwidth_factor = 2.5  # None
    if dropout_scheme == '0':
        missing_mechanism = None
    elif dropout_scheme.startswith('mnar'):
        missing_mechanism = 'mnar'
        if env_class.startswith('linear2d'):
            state_dim = 2
            instrument_var_index = 1 if not misspec_IV else 0  # 1

            if not misspec_MOD:
                if dropout_scheme in ['mnar.v0', 'mnar.v1', 'mnar.v3']:
                    psi_true = 1.5

                    # inputs is a concatenation of [next_states, rewards]
                    def mnar_y_transform(inputs):
                        rewards = inputs[:, [state_dim]]
                        return rewards
                elif dropout_scheme == 'mnar.v2':
                    psi_true = [1, 0.1]
                    initialize_with_psiT = True

                    def mnar_y_transform(inputs):
                        rewards = inputs[:, [state_dim]]
                        return np.hstack([rewards, rewards**2])
            else:
                if dropout_scheme in ['mnar.v0', 'mnar.v1']:

                    def mnar_y_transform(inputs):
                        return inputs
                elif dropout_scheme == 'mnar.v2':

                    def mnar_y_transform(inputs):
                        rewards = inputs[:, [state_dim]]
                        return rewards

            if dropout_scheme in ['mnar.v0', 'mnar.v3']:
                bandwidth_factor = 7.5
            elif dropout_scheme == 'mnar.v1':
                bandwidth_factor = 2  # 2.5
            elif dropout_scheme == 'mnar.v2':
                bandwidth_factor = 7.5 # 7.5  # 2.5
    else:
        missing_mechanism = 'mar'
        parametric_missing_prob = True  # override
    if missing_mechanism is None:
        # no missingness, hence no need of adjustment
        ipw = False
        estimate_missing_prob = False

    # filename suffix configuration
    if not ipw:
        weighting_method = 'cc'
    elif ipw and estimate_missing_prob:
        weighting_method = 'ipw_para_prop' if parametric_missing_prob else 'ipw_semipara_prop'
    elif ipw and not estimate_missing_prob:
        weighting_method = 'ipw_propT'
    if dropout_scheme.startswith('mnar') and estimate_missing_prob:
        if misspec_IV:
            weighting_method += '_misIV'
        if misspec_MOD:
            weighting_method += '_misMOD'
    folder_suffix = ''
    # folder_suffix += f'_missing{dropout_rate}'  # add here
    export_dir = os.path.join(
        log_dir,
        f'{env_class}{folder_suffix}/T{T}_n{n}_policy{policy_id}_gamma{gamma}_dropout_{dropout_scheme}_{weighting_method}'
    )
    pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

    print('Configuration:')
    print('T:', T)
    print(f'n:', n)
    print(f'total_N:', total_N)
    print(f'gamma:', gamma)
    print('ope_method:', ope_method)
    print(f'dropout_scheme:', dropout_scheme)
    print(f'ipw:', ipw)
    print(f'estimate_missing_prob:', estimate_missing_prob)
    print(f'eval_policy_mc_size:', eval_policy_mc_size)
    print(f'eval_horizon:', eval_horizon)
    print(f'Logged to folder:', export_dir)
    # print('')

    # environment, initial states and policy configuration
    np.random.seed(seed=eval_seed)
    if env_class == 'linear2d':
        # specify env and env_dropout
        low = -norm.ppf(0.999)  # -np.inf
        high = norm.ppf(0.999)  # np.inf
        state_dim = 2
        num_actions = 2
        default_key = 'a'
        spline_scaler = MinMaxScaler(min_val=low, max_val=high)

        if vectorize_env:
            env = Linear2dVectorEnv(
                num_envs=n,
                T=T,
                dropout_scheme='0',
                dropout_rate=0.,
                dropout_obs_count_thres=dropout_obs_count_thres,
                low=low,
                high=high)
            env_dropout = Linear2dVectorEnv(
                num_envs=n,
                T=T,
                dropout_scheme=dropout_scheme,
                dropout_rate=dropout_rate,
                dropout_obs_count_thres=dropout_obs_count_thres,
                low=low,
                high=high)
        else:
            env = Linear2dEnv(T=T,
                              dropout_scheme='0',
                              dropout_rate=0.,
                              dropout_obs_count_thres=dropout_obs_count_thres,
                              low=low,
                              high=high)
            env_dropout = Linear2dEnv(
                T=T,
                dropout_scheme=dropout_scheme,
                dropout_rate=dropout_rate,
                dropout_obs_count_thres=dropout_obs_count_thres,
                low=low,
                high=high)

        # eval_S_inits
        eval_S_inits = np.random.normal(loc=0,
                                        scale=1,
                                        size=(eval_policy_mc_size, env.dim))
        eval_S_inits = np.clip(eval_S_inits, a_min=low, a_max=high)
        eval_S_inits_dict = {default_key: eval_S_inits}

        ## specify behavior policy
        behavior_policy = lambda S: env_dropout.action_space.sample()

        ## specify target policy
        assert policy_id in '0123'
        if policy_id == '0':

            def target_policy(S):
                if S[0] + S[1] > 0:
                    return [0, 1]
                else:
                    return [1, 0]

            def vec_target_policy(S):
                if len(S.shape) == 1:
                    S = S.reshape(1, -1)
                return np.where(
                    (S[:, 0] + S[:, 1] > 0).reshape(-1, 1),
                    np.repeat([[0, 1]], repeats=S.shape[0], axis=0),
                    np.repeat([[1, 0]], repeats=S.shape[0], axis=0))
        elif policy_id == '1':  # behavior policy

            def target_policy(S):
                return [0.5, 0.5]

            def vec_target_policy(S):
                if len(S.shape) == 1:
                    S = S.reshape(1, -1)
                return np.repeat([[0.5, 0.5]], repeats=S.shape[0], axis=0)

        elif policy_id == '2':

            def target_policy(S):
                p = np.exp(-S[0] - S[1])
                return [1 / (1 + p), p / (1 + p)]

            def vec_target_policy(S):
                if len(S.shape) == 1:
                    S = S.reshape(1, -1)
                p = np.exp(-S[:, 0] - S[:, 1]).reshape(-1, 1)
                return np.hstack([1 / (1 + p), p / (1 + p)])

        elif policy_id == '3':

            def target_policy(S):
                p = np.exp(S[0] + S[1])
                return [1 / (1 + p), p / (1 + p)]

            def vec_target_policy(S):
                if len(S.shape) == 1:
                    S = S.reshape(1, -1)
                p = np.exp(S[:, 0] + S[:, 1]).reshape(-1, 1)
                return np.hstack([1 / (1 + p), p / (1 + p)])

        policy = vec_target_policy if vectorize_env else target_policy
    elif env_class == 'cartpole':
        num_actions = 2
        state_dim = 4
        default_key = 'a'
        noise_std = 0.02
        spline_scaler = MinMaxScaler()  # fit from data

        # dropout_scheme = '0'
        # dropout_rate = 0.
        dropout_obs_count_thres = 2 # 10

        if vectorize_env:
            env = CartPoleVectorEnv(num_envs=n,
                                    T=T,
                                    noise_std=noise_std,
                                    dropout_scheme='0',
                                    dropout_rate=0.)
            env_dropout = CartPoleVectorEnv(
                num_envs=n,
                T=T,
                noise_std=noise_std,
                dropout_scheme=dropout_scheme,
                dropout_rate=dropout_rate,
                dropout_obs_count_thres=dropout_obs_count_thres)
        else:
            env = CartPoleEnv(T=T,
                              noise_std=noise_std,
                              dropout_scheme='0',
                              dropout_rate=0.)
            env_dropout = CartPoleEnv(
                T=T,
                noise_std=noise_std,
                dropout_scheme=dropout_scheme,
                dropout_rate=dropout_rate,
                dropout_obs_count_thres=dropout_obs_count_thres)

        # eval_S_inits
        # eval_S_inits = np.random.uniform(low=-0.05,
        #                                  high=0.05,
        #                                  size=(eval_policy_mc_size, env.dim))
        # eval_S_inits = np.random.uniform(low=-1,
        #                                  high=1,
        #                                  size=(eval_policy_mc_size, env.dim))
        eval_S_inits = np.random.uniform(low=[-1,-2,-3,-7],
                                         high=[1,2,3,7],
                                         size=(eval_policy_mc_size, env.dim))
        # eval_S_inits = np.random.uniform(low=[-0.5,-1,-2,-3],
        #                                  high=[0.5, 1, 2, 3],
        #                                  size=(eval_policy_mc_size, env.dim))
        eval_S_inits_dict = {default_key: eval_S_inits}

        # train optimal policy using DQN
        rl_env = CartPoleEnv(T=T,
                             noise_std=noise_std,
                             dropout_scheme='0',
                             dropout_rate=0.)
        rl_agent_name = 'dqn'
        rl_model_path = os.path.join(
            log_dir, f"{env_class}{folder_suffix}/{rl_agent_name}_{env_class}")
        print('rl_model_path', rl_model_path)
        if os.path.exists(rl_model_path + '.zip'):
            print('Load optimal policy...')
            rl_agent = DQN.load(path=rl_model_path)
            print('Finished!')
        else:
            print('Learn optimal policy...')
            # rl_env = DefaultCartPoleEnv() # gym.make("CartPole-v1")
            rl_agent = DQN(
                policy="MlpPolicy",
                env=rl_env,
                buffer_size=int(1e5),
                batch_size=64,
                exploration_final_eps=
                0.04,  # final value of random action probability
                exploration_fraction=
                0.16,  # fraction of entire training period over which the exploration rate is reduced
                gradient_steps=
                128,  # How many gradient steps to do after each rollout
                learning_rate=0.005,
                learning_starts=
                1000,  # how many steps of the model to collect transitions for before learning starts
                policy_kwargs=dict(net_arch=[256, 256]),
                target_update_interval=10,
                train_freq=256,  # Update the model every train_freq steps
                verbose=0,
                tensorboard_log=
                f"./logs/{rl_agent_name}_{env_class}_tensorboard/")
            rl_agent.learn(total_timesteps=5e4)
            rl_agent.save(path=rl_model_path)
            print('Finished!')
        # evaluate
        # mean_reward, std_reward = evaluate_policy(model=rl_agent, env=rl_agent.get_env(), n_eval_episodes=30)
        mean_reward, std_reward = evaluate_policy(model=rl_agent,
                                                  env=Monitor(rl_env),
                                                  n_eval_episodes=30,
                                                  deterministic=True)
        print('mean_reward: {:.3f}, std_reward: {:.3f}'.format(
            mean_reward, std_reward))
        print()

        ## specify behavior policy
        # uniform policy
        # behavior_policy = lambda S: env_dropout.action_space.sample()
        def behavior_policy(S):
            if len(S.shape) == 1:
                S = S.reshape(1, -1)
            action_prob =  np.ones((S.shape[0], 2)) * 0.5
            return action_prob

        # # epsilon-greedy
        # epsilon = 0.3 # {0.1,0.2,0.3}
        # def behavior_policy(S):
        #     if len(S.shape) == 1:
        #         S = S.reshape(1, -1)
        #     opt_action_prob = np.zeros((S.shape[0], 2))
        #     q_values = rl_agent.q_net(torch.FloatTensor(S)).detach().numpy()
        #     opt_action_prob[range(S.shape[0]), q_values.argmax(axis=1)] = 1
        #     action_prob = (1 - epsilon) * opt_action_prob + epsilon * (np.ones((S.shape[0], 2)) * 0.5)
        #     return action_prob

        # specify target policy
        assert policy_id in '01'
        if policy_id == '0':

            def vec_target_policy(S):
                # action, _ = rl_agent.predict(S)
                q_values = rl_agent.q_net(torch.FloatTensor(S))
                action_prob = F.softmax(input=q_values, dim=1)
                return action_prob.detach().numpy()
        elif policy_id == '1':

            def vec_target_policy(S):
                # action, _ = rl_agent.predict(S)
                temperature = 10
                q_values = rl_agent.q_net(torch.FloatTensor(S))
                action_prob = F.softmax(input=q_values / temperature, dim=1)
                return action_prob.detach().numpy()

        policy = vec_target_policy
    else:
        raise NotImplementedError

    # get true value via MC approximation (use as baseline)
    true_value_path = os.path.join(
        log_dir,
        f'{env_class}{folder_suffix}/{env_class}_true_value_T{eval_horizon}_policy{policy_id}_gamma{gamma}_size{eval_policy_mc_size}'
    )
    if not os.path.exists(true_value_path):
        print(f'compute true value via MC approximation...')
        agent = SimulationBase(env=env_dropout,
                               n=n,
                               horizon=T + burn_in,
                               discount=gamma,
                               eval_env=env)
        true_value_dict = {}
        for k in eval_S_inits_dict.keys():
            _, true_value_list, true_value_int = agent.evaluate_policy(
                policy=policy,
                seed=eval_seed,
                S_inits=eval_S_inits_dict[k],
                eval_size=eval_policy_mc_size,
                eval_horizon=eval_horizon,
                repeats=int(2e5) //
                eval_policy_mc_size,  # int(2e5) // eval_policy_mc_size
            )
            true_value_dict[k] = {
                'initial_states': eval_S_inits_dict[k],
                'true_value_list': true_value_list,
                'true_value_int': true_value_int,
            }
            print(
                f'initial scenario {k}: true integrated value={true_value_int}'
            )
        with open(true_value_path, 'wb') as outfile:
            pickle.dump(true_value_dict, outfile)
    else:
        with open(true_value_path, 'rb') as outfile:
            true_value_dict = pickle.load(outfile)

    # main part for value estimation simulation
    verbose = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    value_est_summary = {}
    if ope_method == 'mwl':
        value_est_summary_filename = f'{ope_method}_{omega_func_class}_value_est_summary'
    elif ope_method == 'fqe':
        value_est_summary_filename = f'{ope_method}_{Q_func_class}_value_est_summary'
    else:
        value_est_summary_filename = f'{ope_method}_value_est_summary'
    value_est_summary_path = os.path.join(export_dir,
                                          value_est_summary_filename)
    for initial_key in eval_S_inits_dict.keys():
        value_est_list = []
        value_interval_list = collections.defaultdict(list)
        true_value_int = true_value_dict[initial_key]['true_value_int']

        tracking_info = collections.defaultdict(list)
        for itr in range(mc_size):
            seed = itr
            suffix = f'{weighting_method}_itr_{itr}'
            # np.random.seed(seed)
            # if the observational space of the environemnt is bounded, the initial states will only be sampled from uniform distribution
            # if we still want a normal distribution, pass random initial states manually.
            if env_class == 'linear2d':
                train_S_inits = np.random.normal(loc=0,
                                                 scale=1,
                                                 size=(n, env.dim))
                train_S_inits = np.clip(train_S_inits, a_min=low, a_max=high)
            else:
                train_S_inits = None

            # generate data
            if ope_method == 'mwl':
                agent = MWL(env=env_dropout,
                            n=n,
                            horizon=T + burn_in,
                            discount=gamma,
                            eval_env=env,
                            device=device,
                            seed=seed)
            elif ope_method == 'dualdice':
                agent = DualDice(env=env_dropout,
                                 n=n,
                                 horizon=T + burn_in,
                                 discount=gamma,
                                 eval_env=env,
                                 device=device,
                                 seed=seed)
            elif ope_method == 'neuraldice':
                agent = NeuralDice(env=env_dropout,
                                   n=n,
                                   horizon=T + burn_in,
                                   discount=gamma,
                                   eval_env=env,
                                   device=device,
                                   seed=seed)
            elif ope_method == 'fqe':
                agent = FQE(env=env_dropout,
                            n=n,
                            horizon=T + burn_in,
                            discount=gamma,
                            eval_env=env,
                            device=device,
                            seed=seed)
            elif ope_method == 'lstdq':
                agent = LSTDQ(env=env_dropout,
                              n=n,
                              horizon=T + burn_in,
                              discount=gamma,
                              eval_env=env)
            elif ope_method == 'mql':
                agent = MQL(env=env_dropout,
                            n=n,
                            horizon=T + burn_in,
                            discount=gamma,
                            eval_env=env,
                            device=device,
                            seed=seed)
            elif ope_method == 'drl':
                agent = DRL(env=env_dropout,
                            n=n,
                            horizon=T + burn_in,
                            discount=gamma,
                            eval_env=env,
                            device=device,
                            seed=seed)
            agent.dropout_obs_count_thres = max(
                dropout_obs_count_thres - 1, 0)  # -1 because this is the index
            agent.gen_masked_buffer(policy=behavior_policy, # agent.obs_policy,
                                    S_inits=None,
                                    total_N=total_N,
                                    burn_in=burn_in,
                                    seed=seed)
            print(f'dropout rate: {agent.dropout_rate}')
            print(f'missing rate: {agent.missing_rate}')
            print('average length: {}'.format(np.mean([mb[3] for mb in agent.masked_buffer.values()])))
            print('median length: {}'.format(np.median([mb[3] for mb in agent.masked_buffer.values()])))
            print(f'n: {agent.n}')
            print(f'total_N: {agent.total_N}')

            # print(agent.masked_buffer[0][1][:10])
            cum_rewards_list = []
            for i in agent.masked_buffer.keys():
                rewards = agent.masked_buffer[i][2]
                cum_rewards = sum(gamma**t * rewards[t]
                                  for t in range(len(rewards)))
                cum_rewards_list.append(cum_rewards)
            print('empirical value:', np.mean(cum_rewards_list))

            # estimate dropout probability
            if estimate_missing_prob:
                model_suffix = suffix
                # pathlib.Path(os.path.join(export_dir,
                #                           'models')).mkdir(parents=True,
                #                                            exist_ok=True)
                print(f'Fit dropout model ({missing_mechanism})')
                fit_dropout_start = time.time()
                agent.train_dropout_model(
                    model_type=dropout_model_type,
                    missing_mechanism=missing_mechanism,
                    train_ratio=0.8,
                    scale_obs=False,
                    dropout_obs_count_thres=dropout_obs_count_thres,
                    export_dir=os.path.join(export_dir, 'models'),
                    pkl_filename=
                    None,  # f'dropout_model_{dropout_model_type}_T{T}_n{n}_gamma{gamma}_{model_suffix}.pkl'
                    seed=seed,
                    include_reward=
                    False,  # [True, False], if True, will override function mnar_y_transform()
                    instrument_var_index=instrument_var_index,
                    mnar_y_transform=mnar_y_transform,
                    psi_init=None if missing_mechanism == 'mnar'
                    and not initialize_with_psiT else psi_true,
                    parametric=parametric_missing_prob,
                    bandwidth_factor=bandwidth_factor,
                    verbose=True)
                print(f'Estimate dropout propensities')
                agent.estimate_missing_prob(
                    missing_mechanism=missing_mechanism)
                fit_dropout_end = time.time()
                print(
                    f'Finished! {fit_dropout_end-fit_dropout_start} secs elapsed.'
                )

            # estimate value
            if ope_method == 'mwl':  # minimax weight learning
                if omega_func_class == 'nn':
                    agent.estimate_omega(
                        target_policy=DiscretePolicy(policy_func=policy,
                                                     num_actions=num_actions),
                        initial_state_sampler=InitialStateSampler(
                            initial_states=eval_S_inits_dict[initial_key],
                            seed=seed),
                        omega_func_class='nn',
                        Q_func_class='rkhs',
                        hidden_sizes=[64, 64],
                        separate_action=True,  # True, False
                        max_iter=1000,
                        batch_size=32,
                        lr=0.0005,
                        ipw=ipw,
                        estimate_missing_prob=estimate_missing_prob,
                        prob_lbound=prob_lbound,
                        print_freq=50,
                        scaler='Standard',
                        patience=100,  # 10
                        verbose=verbose)
                    # kwargs = {'q_lr': 1e-4, 'omega_lr': 1e-4}
                    # agent.estimate_omega(
                    #     target_policy=DiscretePolicy(policy_func=policy,
                    #                                  num_actions=num_actions),
                    #     initial_state_sampler=InitialStateSampler(
                    #         initial_states=eval_S_inits_dict[initial_key],
                    #         seed=seed),
                    #     omega_func_class='nn',
                    #     hidden_sizes=[64, 64],
                    #     separate_action=True,  # True, False
                    #     max_iter=2000,
                    #     batch_size=1024,
                    #     # lr=0.0005,
                    #     ipw=ipw,
                    #     estimate_missing_prob=estimate_missing_prob,
                    #     prob_lbound=prob_lbound,
                    #     print_freq=50,
                    #     scaler='Standard',
                    #     # patience=100,
                    #     verbose=verbose)
                elif omega_func_class == 'spline':
                    agent.estimate_omega(
                        target_policy=DiscretePolicy(policy_func=policy,
                                                     num_actions=num_actions),
                        initial_state_sampler=InitialStateSampler(
                            initial_states=eval_S_inits_dict[initial_key],
                            seed=seed),
                        omega_func_class=omega_func_class,
                        Q_func_class=Q_func_class,
                        ipw=ipw,
                        estimate_missing_prob=estimate_missing_prob,
                        prob_lbound=prob_lbound,
                        scaler=MinMaxScaler(min_val=env.low, max_val=env.high)
                        if env is not None else MinMaxScaler(),
                        # spline fitting related arguments
                        ridge_factor=ridge_factor,
                        L=max(spline_degree + 1, dof),
                        d=spline_degree,
                        knots=knots,
                        product_tensor=product_tensor,
                        basis_scale_factor=basis_scale_factor,
                        verbose=verbose)
                elif omega_func_class == 'expo_linear':
                    agent.estimate_omega(
                        target_policy=DiscretePolicy(policy_func=policy,
                                                     num_actions=num_actions),
                        initial_state_sampler=InitialStateSampler(
                            initial_states=eval_S_inits_dict[initial_key],
                            seed=seed),
                        omega_func_class='expo_linear',
                        Q_func_class=Q_func_class,
                        ipw=ipw,
                        estimate_missing_prob=estimate_missing_prob,
                        prob_lbound=prob_lbound,
                        scaler=MinMaxScaler(min_val=env.low, max_val=env.high)
                        if env is not None else MinMaxScaler(),
                        L=max(spline_degree + 1, dof),
                        d=spline_degree,
                        knots=knots,
                        product_tensor=product_tensor,
                        basis_scale_factor=
                        3,  # to ensure the input lies in a reasonable range, otherwise the training is not as stable
                        lr=5e-3,
                        batch_size=256,
                        max_iter=3000,  # 2000
                        print_freq=50,
                        patience=100,
                        verbose=verbose)

                value_est = agent.get_value()
                if mc_size <= 2:
                    print('value true: {:.3f}'.format(true_value_int))
                    print('value est: {:.3f}'.format(value_est))
                    agent.validate_visitation_ratio(grid_size=10,
                                                    visualize=True,
                                                    quantile=vis_quantile,
                                                    prefix='{}_'.format(env_class))
            elif ope_method == 'dualdice':
                zeta_pos = True
                nu_network_kwargs = {
                    'hidden_sizes': [64, 64],
                    'hidden_nonlinearity': nn.ReLU(),
                    'hidden_w_init': nn.init.xavier_uniform_,
                    'output_w_inits': nn.init.xavier_uniform_
                }
                output_activation_fn = Square() if zeta_pos else nn.Identity()
                zeta_network_kwargs = {
                    'hidden_sizes': [64, 64],
                    'hidden_nonlinearity': nn.ReLU(),
                    'hidden_w_init': nn.init.xavier_uniform_,
                    'output_w_inits': nn.init.xavier_uniform_,
                    'output_nonlinearity': output_activation_fn
                }

                agent.estimate_omega(
                    target_policy=DiscretePolicy(policy_func=policy,
                                                 num_actions=num_actions),
                    initial_state_sampler=InitialStateSampler(
                        initial_states=eval_S_inits_dict[initial_key],
                        seed=seed),
                    nu_network_kwargs=nu_network_kwargs,
                    nu_learning_rate=3e-4,  # 1e-4
                    zeta_network_kwargs=zeta_network_kwargs,
                    zeta_learning_rate=3e-4,  # 1e-4
                    zeta_pos=zeta_pos,
                    solve_for_state_action_ratio=True,
                    ipw=ipw,
                    estimate_missing_prob=estimate_missing_prob,
                    prob_lbound=prob_lbound,
                    max_iter=15000,
                    batch_size=1024,
                    f_exponent=2,
                    primal_form=False,
                    scaler="Standard",
                    print_freq=500,
                    verbose=verbose)
                value_est = agent.get_value()
                if mc_size <= 2:
                    print('value est: {:.3f}'.format(value_est))
                    agent.validate_visitation_ratio(grid_size=10,
                                                    visualize=True,
                                                    quantile=vis_quantile,
                                                    prefix='{}_'.format(env_class))
            elif ope_method == 'neuraldice':
                zeta_pos = True
                zero_reward = True
                primal_regularizer = 0.
                dual_regularizer = 1.
                norm_regularizer = 1.
                nu_regularizer = 0.
                zeta_regularizer = 0.
                nu_learning_rate = 1e-4  # 1e-4
                zeta_learning_rate = 1e-4  # 1e-4

                nu_network = QNetwork(input_dim=state_dim,
                                      output_dim=num_actions,
                                      hidden_sizes=[64, 64],
                                      hidden_nonlinearity=nn.ReLU(),
                                      hidden_w_init=nn.init.xavier_uniform_,
                                      output_w_inits=nn.init.xavier_uniform_)
                output_activation_fn = Square() if zeta_pos else nn.Identity()
                zeta_network = QNetwork(
                    input_dim=state_dim,
                    output_dim=num_actions,
                    hidden_sizes=[64, 64],
                    hidden_nonlinearity=nn.ReLU(),
                    hidden_w_init=nn.init.xavier_uniform_,
                    output_w_inits=nn.init.xavier_uniform_,
                    output_nonlinearity=output_activation_fn)

                agent.estimate_omega(
                    target_policy=DiscretePolicy(policy_func=policy,
                                                 num_actions=num_actions),
                    initial_state_sampler=InitialStateSampler(
                        initial_states=eval_S_inits_dict[initial_key],
                        seed=seed),
                    nu_network=nu_network,
                    zeta_network=zeta_network,
                    nu_learning_rate=nu_learning_rate,
                    zeta_learning_rate=zeta_learning_rate,
                    zero_reward=zero_reward,
                    solve_for_state_action_ratio=True,
                    f_exponent=2,
                    primal_form=False,
                    primal_regularizer=primal_regularizer,
                    dual_regularizer=dual_regularizer,
                    norm_regularizer=norm_regularizer,
                    nu_regularizer=nu_regularizer,
                    zeta_regularizer=zeta_regularizer,
                    weight_by_gamma=False,
                    ipw=ipw,
                    estimate_missing_prob=estimate_missing_prob,
                    prob_lbound=prob_lbound,
                    scaler="Standard",
                    max_iter=15000,
                    batch_size=1024,
                    print_freq=500,
                    verbose=verbose)
                value_est = agent.get_value()
                if mc_size <= 2:
                    print('value est: {:.3f}'.format(value_est))
                    agent.validate_visitation_ratio(grid_size=10,
                                                    visualize=True,
                                                    quantile=vis_quantile,
                                                    prefix='{}_'.format(env_class))
            elif ope_method == 'fqe':  # fitted Q-evalution
                if Q_func_class == 'nn':
                    agent.estimate_Q(
                        target_policy=DiscretePolicy(policy_func=policy,
                                                     num_actions=num_actions),
                        ipw=ipw,
                        estimate_missing_prob=estimate_missing_prob,
                        prob_lbound=prob_lbound,
                        max_iter=100,
                        tol=0.01,
                        func_class=Q_func_class,
                        scaler='Standard',
                        verbose=verbose,
                        hidden_sizes=[64, 64],
                        lr=0.001,
                        batch_size=128,
                        epoch=50,
                        patience=10,
                        print_freq=10)
                elif Q_func_class == 'rf':
                    agent.estimate_Q(
                        target_policy=DiscretePolicy(policy_func=policy,
                                                     num_actions=num_actions),
                        ipw=ipw,
                        estimate_missing_prob=estimate_missing_prob,
                        prob_lbound=prob_lbound,
                        max_iter=250,
                        tol=0.001,
                        func_class=Q_func_class,
                        scaler='Standard',
                        verbose=verbose,
                        print_freq=10,
                        n_estimators=500,
                        max_depth=15,
                        min_samples_leaf=10)
                if Q_func_class == 'spline':
                    agent.estimate_Q(
                        target_policy=DiscretePolicy(policy_func=policy,
                                                     num_actions=num_actions),
                        ipw=ipw,
                        estimate_missing_prob=estimate_missing_prob,
                        prob_lbound=prob_lbound,
                        max_iter=100,
                        tol=0.001,
                        func_class=Q_func_class,
                        scaler='MinMax',
                        verbose=verbose,
                        print_freq=10,
                        # spline fitting related arguments
                        ridge_factor=
                        ridge_factor,  # recommend using a slightly larger weight than LSTDQ
                        L=max(spline_degree + 1, dof),
                        d=spline_degree,
                        knots=knots,
                        product_tensor=product_tensor,
                        basis_scale_factor=basis_scale_factor)
                print("Getting value estimate...\n")
                value_est = agent.get_value(
                    S_inits=eval_S_inits_dict[initial_key])
                print('value est: {:.3f}'.format(value_est))
                if mc_size <= 2:
                    agent.validate_Q(grid_size=10,
                                     visualize=True,
                                     quantile=vis_quantile,
                                     prefix='{}_'.format(env_class))
            elif ope_method == 'lstdq':
                assert Q_func_class == 'spline'
                if env_class == 'cartpole': 
                    spline_scaler.reset()
                agent.estimate_Q(target_policy=policy,
                                 ipw=ipw,
                                 estimate_missing_prob=estimate_missing_prob,
                                 weight_curr_step=weight_curr_step,
                                 prob_lbound=prob_lbound,
                                 ridge_factor=ridge_factor,
                                 L=max(spline_degree + 1, dof),
                                 d=spline_degree,
                                 knots=knots,
                                 scaler=spline_scaler,
                                 product_tensor=product_tensor,
                                 basis_scale_factor=basis_scale_factor,
                                 grid_search=False,
                                 verbose=verbose)
                print("Getting value estimate...")
                value_est = agent.get_value(
                    S_inits=eval_S_inits_dict[initial_key])
                print("Getting value interval estimate...")
                value_interval_est = agent.get_value_interval(
                    S_inits=eval_S_inits_dict[initial_key],
                    alpha_list=CI_alphas)
                print('value true: {:.3f}'.format(true_value_int))
                print('value est: {:.3f}'.format(value_est))
                print('value interval:', value_interval_est)
                if mc_size <= 2:
                    agent.validate_Q(grid_size=10,
                                     visualize=True,
                                     seed=seed,
                                     quantile=vis_quantile,
                                     prefix='{}_'.format(env_class),
                                     slice_dim=(0,1), # (0,2)
                                     mark_eval_states=True, 
                                     eval_states=eval_S_inits_dict[initial_key])  # sanity check
                print()  # delimiter
            elif ope_method == 'mql':
                agent.estimate_Q(target_policy=DiscretePolicy(
                    policy_func=policy, num_actions=num_actions),
                                 max_iter=500,
                                 hidden_sizes=[64, 64],
                                 lr=0.0005,
                                 batch_size=64,
                                 target_update_frequency=100,
                                 patience=50,
                                 scaler='Standard',
                                 print_freq=50,
                                 verbose=verbose)
                value_est = agent.get_value(
                    S_inits=eval_S_inits_dict[initial_key])
                print(value_est)
                if mc_size <= 2:
                    agent.validate_Q(grid_size=10,
                                    visualize=True,
                                    quantile=vis_quantile,
                                    prefix='{}_'.format(env_class))
            elif ope_method == 'drl':
                common_kwargs = {
                    'env': env_dropout,
                    'n': n,
                    'horizon': T + burn_in,
                    'discount': gamma,
                    'eval_env': env,
                    'device': device,
                    'seed': seed
                }
                estimate_omega_kwargs = {
                    'func_class':
                    'spline',
                    'ipw':
                    ipw,
                    'estimate_missing_prob':
                    estimate_missing_prob,
                    'prob_lbound':
                    prob_lbound,
                    'scaler':
                    MinMaxScaler(min_val=env.low, max_val=env.high)
                    if env is not None else MinMaxScaler(),
                    'ridge_factor':
                    ridge_factor,
                    'L':
                    max(spline_degree + 1, dof),
                    'd':
                    spline_degree,
                    'knots':
                    knots,
                    'product_tensor':
                    product_tensor,
                    'basis_scale_factor':
                    basis_scale_factor,
                    'verbose':
                    verbose
                }
                if Q_func_class == 'rf':
                    estimate_Q_kwargs = {
                        'max_iter': 200,
                        'tol': 0.001,
                        'func_class': 'rf',
                        'scaler': 'Standard',
                        'n_estimators': 250,
                        'max_depth': 15,
                        'min_samples_leaf': 10,
                        'verbose': verbose
                    }
                elif Q_func_class == 'spline':
                    estimate_Q_kwargs = {
                        'ipw': ipw,
                        'estimate_missing_prob': estimate_missing_prob,
                        'weight_curr_step': weight_curr_step,
                        'prob_lbound': prob_lbound,
                        'ridge_factor': ridge_factor,
                        'L': max(spline_degree + 1, dof),
                        'd': spline_degree,
                        'knots': knots,
                        'scale': 'MinMax',
                        'product_tensor': product_tensor,
                        'basis_scale_factor': basis_scale_factor,
                        'grid_search': False,
                        'verbose': verbose
                    }

                omega_estimator = MWL(**common_kwargs)
                Q_estimator = LSTDQ(
                    **common_kwargs) if Q_func_class == 'spline' else FQE(
                        **common_kwargs)
                print('Fitting model for omega...')
                agent.estimate_omega(
                    omega_estimator=omega_estimator,
                    target_policy=DiscretePolicy(policy_func=policy,
                                                 num_actions=num_actions),
                    initial_state_sampler=InitialStateSampler(
                        initial_states=eval_S_inits_dict[initial_key],
                        seed=seed),
                    estimate_omega_kwargs=estimate_omega_kwargs)
                print('\nFitting model for Q-function...')
                agent.estimate_Q(Q_estimator=Q_estimator,
                                 target_policy=DiscretePolicy(
                                     policy_func=policy,
                                     num_actions=num_actions),
                                 estimate_Q_kwargs=estimate_Q_kwargs)
                print("Getting value estimate...")
                value_est = agent.get_value()
                print("Getting value interval estimate...\n")
                value_interval_est = agent.get_value_interval(
                    alpha_list=CI_alphas)
                if mc_size <= 2:
                    print('value est: {:.3f}'.format(value_est))
                    print('value interval: {}\n'.format(value_interval_est))
                    agent.validate_Q(grid_size=10,
                                     visualize=True,
                                     quantile=vis_quantile,
                                     prefix='{}_'.format(env_class))
                    agent.validate_visitation_ratio(grid_size=10,
                                                    visualize=True,
                                                    quantile=vis_quantile,
                                                    prefix='{}_'.format(env_class))
            else:
                raise NotImplementedError

            value_est_list.append(value_est)
            value_est_summary[initial_key] = {
                'initial_states': eval_S_inits_dict[initial_key],
                'value_est_list': value_est_list,
                'true_value_int': true_value_int
            }
            if ope_method in ['lstdq', 'drl']:
                for alpha in CI_alphas:
                    value_interval_list[alpha].append(
                        (value_interval_est['lower_bound'][alpha],
                         value_interval_est['upper_bound'][alpha]))
                value_est_summary[initial_key][
                    'value_interval_list'] = value_interval_list

            # if tracking_info:
            #     print(tracking_info)

            with open(value_est_summary_path, 'wb') as outfile:
                pickle.dump(value_est_summary, outfile)

            # if ope_method == 'mwl':
            #     # decompose error of value estimation into 3 parts
            #     # step 1: approximate the true Q-function
            #     q_model = LSTDQ(env=env_dropout,
            #                     n=n, # 5000
            #                     horizon=T + burn_in,
            #                     discount=gamma,
            #                     eval_env=env)
            #     q_model.dropout_obs_count_thres = max(
            #         dropout_obs_count_thres - 1, 0)  # -1 because this is the index
            #     q_model.gen_masked_buffer(policy=behavior_policy, # q_model.obs_policy,
            #                             S_inits=None,
            #                             total_N=total_N,
            #                             burn_in=burn_in,
            #                             seed=seed)
            #     q_model.estimate_Q(target_policy=policy,
            #                        ipw=ipw,
            #                        estimate_missing_prob=estimate_missing_prob,
            #                        weight_curr_step=weight_curr_step,
            #                        prob_lbound=prob_lbound,
            #                        ridge_factor=ridge_factor,
            #                        L=max(spline_degree+1, dof),
            #                        d=spline_degree,
            #                        knots=knots,
            #                        scaler='MinMax',
            #                        product_tensor=product_tensor,
            #                        basis_scale_factor=basis_scale_factor,
            #                        grid_search=False,
            #                        verbose=verbose)
            #     value_baseline = q_model.get_value(S_inits=eval_S_inits_dict[initial_key])
            #     print('baseline value:', value_baseline)
            #     # parr 1:
            #     state = agent.replay_buffer.states
            #     action = agent.replay_buffer.actions
            #     next_state = agent.replay_buffer.next_states
            #     reward = agent.replay_buffer.rewards
            #     initial_state = eval_S_inits_dict[initial_key]
            #     omega_hat = agent.omega(states=state, actions=action)
            #     q_pred = q_model.Q(states=next_state, actions=action)
            #     q_next_pred = q_model.V(states=next_state, policy=policy).flatten()
            #     pseudo_resid = q_pred - gamma * q_next_pred

            #     error_pt1 = np.mean(omega_hat * pseudo_resid, axis=0) - \
            #         (1 - gamma) * np.mean(q_model.V(states=initial_state, policy=policy), axis=0)
            #     error_pt1 = error_pt1.item() / (1 - gamma)
            #     print('error part 1:', error_pt1)

            #     def denoised_reward_function(inputs):
            #         state = inputs[:,:-1]
            #         action = inputs[:,[-1]]
            #         raw_reward = (2 * state[:,0] - state[:,1] - 0.25) * (2 * action - 1) + 0.5 * state[:,1]
            #         return raw_reward

            #     inputs = np.hstack([state, action.reshape(-1,1)])
            #     epsilon = reward - denoised_reward_function(inputs)
            #     error_pt2 = np.mean(omega_hat * epsilon) / (1 - gamma)
            #     print('error part 2:', error_pt2)

            #     expand_dim_resid = reward + gamma * q_next_pred - q_pred - epsilon
            #     error_pt3 = np.mean(omega_hat * expand_dim_resid) / (1 - gamma)
            #     print('error part 3:', error_pt3)

            #     print('bias:', value_est - value_baseline)
            #     tracking_info['error_pt1'].append(error_pt1)
            #     tracking_info['error_pt2'].append(error_pt2)
            #     tracking_info['error_pt3'].append(error_pt3)
            #     tracking_info['bias'].append(value_est - value_baseline)

        if mc_size > 1:
            abnormal_thres = 10 # float('inf')
            abnormal_index = []
            if ope_method in ['lstdq', 'drl']:
                als = [u - l for l, u in value_interval_list[CI_alphas[0]]]
                abnormal_index = [i for i in range(len(als)) if als[i] > 10]
            value_est_list = [v for i, v in enumerate(value_est_list) if i not in abnormal_index]
            value_interval_list[CI_alphas[0]] = [v for i, v in enumerate(value_interval_list[CI_alphas[0]]) if i not in abnormal_index]

            print(f'Summary:')
            print(f'[initial state scheme]', initial_key)
            print(f'[true V_int] {true_value_int}')
            output_str = '[est V_int] average: {:.3f}'.format(
                np.mean(value_est_list)) + ' std: {:.3f}'.format(
                    np.std(value_est_list, ddof=1)) + ' bias: {:.3f}'.format(
                        np.mean(value_est_list) -
                        true_value_int) + ' RMSE: {:.3f}'.format(
                            np.mean((np.array(value_est_list) - true_value_int)
                                    **2))
            # print(
            #     '[est V_int] average: {:.3f}'.format(np.mean(value_est_list)),
            #     'std: {:.3f}'.format(np.std(value_est_list, ddof=1)),
            #     'bias: {:.3f}'.format(
            #         np.mean(value_est_list) - true_value_int),
            #     'RMSE: {:.3f}'.format(
            #         np.mean((np.array(value_est_list) - true_value_int)**2)))
            if ope_method in ['lstdq', 'drl']:
                ecp = np.mean([
                    l <= true_value_int <= u
                    for l, u in value_interval_list[CI_alphas[0]]
                ])
                als = [u - l for l, u in value_interval_list[CI_alphas[0]]]
                al_mean = np.mean(als)
                al_std = np.std(als, ddof=1)
                output_str += ' ECP: {:.3f}'.format(ecp) + ' AL: {:.3f} ({:.3f})'.format(al_mean, al_std)
                # print('ECP: {:.3f}'.format(ecp), 'AL: {:.3f}'.format(al))
            print(output_str)

        value_est_summary[initial_key] = {
            'initial_states': eval_S_inits_dict[initial_key],
            'value_est_list': value_est_list,
            'true_value_int': true_value_int
        }
        if ope_method in ['lstdq', 'drl']:
            value_est_summary[initial_key][
                'value_interval_list'] = value_interval_list

        # if tracking_info:
        #     print(tracking_info)

    with open(value_est_summary_path, 'wb') as outfile:
        pickle.dump(value_est_summary, outfile)
