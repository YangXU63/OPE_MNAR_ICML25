"""
Examine the point-estimator of the integrated value.
"""

import os
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.stats import norm
import argparse
import pathlib
import time
import gc

try:
    from custom_env.linear2d import Linear2dEnv, Linear2dVectorEnv
    from ope_mnar.main import train_Q_func, get_target_value_multi
except:
    import sys
    sys.path.append(os.path.expanduser('~/ope_mnar/ope_mnar'))
    sys.path.append(os.path.expanduser('~/ope_mnar'))
    from custom_env.linear2d import Linear2dEnv, Linear2dVectorEnv
    from ope_mnar.main import train_Q_func, get_target_value_multi

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='linear2d')
parser.add_argument('--max_episode_length', type=int, default=25)  # 10, 25
parser.add_argument('--discount', type=float, default=0.8)
parser.add_argument('--num_trajs', type=int, default=500)  # 250, 500
parser.add_argument('--burn_in', type=int, default=0)
parser.add_argument('--mc_size', type=int, default=1) 
parser.add_argument('--eval_policy_mc_size', type=int, default=100) # 10000 might be too big
parser.add_argument('--eval_horizon', type=int, default=250)
parser.add_argument('--dropout_scheme', type=str, default='mnar.v0', choices=["0", "mnar.v0", "mar.v0"])  # 'mnar.v0'
parser.add_argument('--dropout_rate', type=float, default=0.9)
parser.add_argument('--dropout_model', type=str, default='semiparam') # can take either 'param' or 'semiparam' method
parser.add_argument(
    '--dropout_obs_count_thres',
    type=int,
    default=2,
    help="the number of observations that is not subject to dropout")
parser.add_argument('--ipw',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)  # True
parser.add_argument('--estimate_missing_prob',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)  # True
parser.add_argument('--weight_curr_step',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--env_model_type', type=str, default='linear')
parser.add_argument('--vectorize_env',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
args = parser.parse_args()

if __name__ == '__main__':
    log_dir = os.path.expanduser('~/ope_mnar/output')

    env_class = args.env
    T = args.max_episode_length
    n = args.num_trajs
    total_N = None
    gamma = args.discount
    mc_size = args.mc_size
    dropout_scheme = args.dropout_scheme
    dropout_rate = args.dropout_rate
    dropout_model = args.dropout_model
    dropout_obs_count_thres = args.dropout_obs_count_thres
    burn_in = args.burn_in
    vectorize_env = args.vectorize_env
    ipw = args.ipw
    weight_curr_step = args.weight_curr_step  # if False, use survival probability
    estimate_missing_prob = args.estimate_missing_prob
    eval_policy_mc_size = args.eval_policy_mc_size
    eval_horizon = args.eval_horizon
    eval_seed = 123
    prob_lbound = 1e-2
    
    # model configuration
    default_scaler = "MinMax"  # "NormCdf", "MinMax"
    scale = default_scaler
    adaptive_dof = False
    product_tensor = True
    grid_search = False
    basis_scale_factor = 100  # 100
    spline_degree = 3
    if env_class == 'linear2d':
        if adaptive_dof:
            dof = max(4, int(((n * T) ** (3 / 7)) ** (1 / 2)))  # degree of freedom
        else:
            dof = 7
    elif env_class == 'cartpole':
        if adaptive_dof:
            dof = max(4, int(((n * T)**(3/7))**(1/4))) # degree of freedom
        else:
            dof = 4
    ridge_factor = 1e-3  # 1e-3
    if scale == default_scaler:
        knots = np.linspace(start=-spline_degree / (dof - spline_degree),
                            stop=1 + spline_degree / (dof - spline_degree),
                            num=dof + spline_degree + 1)  # handle the boundary
    else:
        knots = 'equivdist'  # 'equivdist', None

    # dropout model configuration
    dropout_model_type = 'linear'
    instrument_var_index = None
    mnar_y_transform = None
    bandwidth_factor = None
    if env_class.lower() == 'linear2d':
        if dropout_scheme == '0':
            missing_mechanism = None
        elif dropout_scheme.startswith('mnar'):
            missing_mechanism = 'mnar'
            instrument_var_index = 1
            if dropout_scheme == 'mnar.v0':
                bandwidth_factor = 7.5
            elif dropout_scheme == 'mnar.v1':
                bandwidth_factor = 2.5
        else:
            missing_mechanism = 'mar'
    psi_true = None  # 1.5, None
    if missing_mechanism and missing_mechanism.lower() == 'mnar':
        initialize_with_psiT = False
    if missing_mechanism is None:
        # no missingness, hence no need of adjustment
        ipw = False
        estimate_missing_prob = False

    # filename suffix configuration
    if not ipw:
        weighting_method = 'cc'
    elif ipw and estimate_missing_prob:
        weighting_method = 'ipw_propF'
    elif ipw and not estimate_missing_prob:
        weighting_method = 'ipw_propT'
    folder_suffix = ''
    folder_suffix += f'_missing{dropout_rate}'
    folder_suffix += f'_ridge{ridge_factor}'
    if basis_scale_factor != 1:
        folder_suffix += f'_scale{int(basis_scale_factor)}'

    export_dir = os.path.join(
        log_dir,
        f'{env_class}_est_value{folder_suffix}/T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout_{dropout_scheme}'
    )
    true_value_dir = os.path.join(log_dir,
                                  f'{env_class}_est_value{folder_suffix}')
    true_value_dir
    pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

    print('Configuration:')
    print(f'T : {T}')
    print(f'n : {n}')
    print(f'total_N : {total_N}')
    print(f'spline degree of freedom : {dof}')
    print(f'gamma : {gamma}')
    print(f'dropout_scheme : {dropout_scheme}')
    print(f'dropout_model : {dropout_model}')
    print(f'ipw : {ipw}')
    print(f'estimate_missing_prob : {estimate_missing_prob}')
    print(f'eval_policy_mc_size : {eval_policy_mc_size}')
    print(f'eval_horizon : {eval_horizon}')
    print(f'Logged to folder: {export_dir}')
    
    np.random.seed(seed=eval_seed)
    default_key = 'C'
    if env_class.lower() == 'linear2d':
        # specify env and env_dropout
        low = -norm.ppf(0.999)  # -np.inf
        high = norm.ppf(0.999)  # np.inf

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
            env = Linear2dEnv(
                T=T,
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

        # specify target policy
        def target_policy(S):
            if S[0] + S[1] > 0:
                return [0, 1]
            else:
                return [1, 0]

        def vec_target_policy(S):
            if len(S.shape) == 1:
                S = S.reshape(1, -1)
            return np.where((S[:, 0] + S[:, 1] > 0).reshape(-1, 1),
                            np.repeat([[0, 1]], repeats=S.shape[0], axis=0),
                            np.repeat([[1, 0]], repeats=S.shape[0], axis=0))
        
        policy = vec_target_policy if vectorize_env else target_policy
    else:
        raise NotImplementedError

    V_est_list = []
    for itr in range(mc_size):
        np.random.seed(itr)
        # if the observational space of the environemnt is bounded, the initial states will only be sampled from uniform distribution
        # if we still want a normal distribution, pass random initial states manually.
        if env_class.lower() == 'linear2d':
            train_S_inits = np.random.normal(loc=0, scale=1, size=(n, env.dim))
            train_S_inits = np.clip(train_S_inits, a_min=low, a_max=high)
        else:
            train_S_inits = None
        if ipw:
            suffix = f'{weighting_method}_itr_{itr}'
        else:
            suffix = f'itr_{itr}'
        filename_train = f'train_with_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout_{dropout_scheme}_{suffix}'
        filename_value_int = f'value_int_with_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout_{dropout_scheme}_{suffix}'
        filename_value = f'value_with_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout_{dropout_scheme}_{suffix}'
        filename_true_value = f'{env_class}_true_value_T_{eval_horizon}_gamma{gamma}_size{eval_policy_mc_size}'

        print('Train Q-function...')
        start = time.time()
        train_Q_func(
            T=T,
            n=n,
            env=env_dropout,
            L=dof,
            d=spline_degree,
            knots=knots,
            total_N=total_N,
            burn_in=burn_in,
            target_policy=policy,
            export_dir=export_dir,
            scale=scale,
            product_tensor=product_tensor,
            discount=gamma,
            seed=itr,
            S_inits=None,  # train_S_inits, None
            ipw=ipw,
            weight_curr_step=weight_curr_step,
            estimate_missing_prob=estimate_missing_prob,
            dropout_obs_count_thres=dropout_obs_count_thres,
            missing_mechanism=missing_mechanism,
            instrument_var_index=instrument_var_index,
            mnar_y_transform=mnar_y_transform,
            psi_init=None if missing_mechanism == 'mnar'
            and not initialize_with_psiT else psi_true,
            bandwidth_factor=bandwidth_factor,
            ridge_factor=ridge_factor,
            grid_search=grid_search,
            basis_scale_factor=basis_scale_factor,
            dropout_model_type=dropout_model_type,
            dropout_model = dropout_model,
            dropout_scale_obs=False,  # True, False
            dropout_include_reward=True,  # True, False
            model_suffix=suffix,
            prob_lbound=prob_lbound,
            eval_env=env,
            filename_train=filename_train)

        end = time.time()
        print('Finished! Elapsed time: %.3f mins' % ((end - start) / 60))

        # evaluate the value
        print('Evaluate target policy...')
        start = time.time()

        value_dict = get_target_value_multi(
            T=T,
            n=n,
            env=env_dropout,
            eval_T=eval_horizon,
            vf_mc_size=eval_policy_mc_size,
            target_policy=policy,
            use_vector_env=vectorize_env,
            import_dir=export_dir,
            filename_train=filename_train,
            filename_true_value=filename_true_value,
            export_dir=export_dir,
            value_import_dir=true_value_dir,
            scale=scale,
            product_tensor=product_tensor,
            discount=gamma,
            eval_env=env,
            dropout_model = dropout_model,
            filename_value=filename_value,
            eval_S_inits_dict=eval_S_inits_dict,
            eval_seed=eval_seed,  # itr
        )
        print('Evaluation Done!!!')

        print({
            'MeanSE':
            value_dict[eval_horizon][default_key]['MeanSE'],
            'actual_value_int':
            value_dict[eval_horizon][default_key]['actual_value_int'],
            'est_value_int':
            value_dict[eval_horizon][default_key]['est_value_int']
        })

        V_est_list.append(
            value_dict[eval_horizon][default_key]['est_value_int'][0])

        end = time.time()
        print('Finished! Elapsed time: %.3f mins \n' % ((end - start) / 60))
        _ = gc.collect()

    print(V_est_list)
    if mc_size > 1:
        print('[est V_int] average:', np.mean(V_est_list), ', std:',
              np.std(V_est_list, ddof=1))
