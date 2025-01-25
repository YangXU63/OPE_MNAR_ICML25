import os
import numpy as np
import pickle
from collections import defaultdict, Counter
import pathlib
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import time

from direct_method import LSTDQ


def train_Q_func(
        T=30,
        n=25,
        env=None,
        L=None,
        d=3,
        knots=None,
        total_N=None,
        burn_in=0,
        target_policy=None,
        export_dir=None,
        scale='NormCdf',
        product_tensor=True,
        discount=0.5,
        seed=None,
        S_inits=None,
        ipw=False,
        weight_curr_step=True,
        estimate_missing_prob=False,
        dropout_obs_count_thres=1,
        missing_mechanism=None,
        instrument_var_index=None,
        mnar_y_transform=None,
        psi_init=None,
        bandwidth_factor=1.5,
        ridge_factor=0.,
        grid_search=False,
        basis_scale_factor=1.,
        dropout_model_type='linear',
        dropout_model = 'semiparam',
        dropout_include_reward=False,
        dropout_scale_obs=False,
        model_suffix='',
        prob_lbound=1e-3,
        eval_env=None,
        filename_train=None):
    """
    Args:
        T (int): maximum horizon of observed trajectories
        n (int): number of observed trajectories
        env (gym.Env): environment
        L (int): number of basis function (degree of freedom)
        d (int): B-spline degree
        knots (str or np.ndarray): location of knots
        total_N (int): total number of state-action pairs
        burn_in (int): length of burn-in period
        use_vector_env (bool): if True, use vectoried environment. This can accelerate the calculation.
        target_policy (callable): target policy to be evaluated
        export_dir (str): directory to export results
        scale (str): scaler to transform state features onto [0,1], 
                select from "NormCdf", "Identity", "MinMax", or a path to a fitted scaler
        product_tensor (bool): if True, use product tensor to construct basis
        discount (float): discount factor
        seed (int): random seed to generate trajectories
        S_inits (np.ndarray): initial states for generated trajectories
        S_inits_kwargs (dict): additional kwargs passed to env.reset()
        ipw (bool): if True, use inverse probability weighting to adjust for missing data
        weight_curr_step (bool): if True, use the probability of dropout at current step
        estimate_missing_prob (bool): if True, use estimated missing probability, otherwise, use ground truth (only for simulation)
        dropout_obs_count_thres (int): number of observations that is not subject to dropout
        missing_mechanism (str): "mnar" or "mar"
        instrument_var_index (int): index of the instrument variable
        mnar_y_transform (callable): input next_obs and reward, output Y term for the mnar dropout model
        psi_init (float): initial value for psi in MNAR estimation
        bandwidth_factor (float): the constant used in bandwidth calculation
        ridge_factor (float): ridge penalty parameter
        grid_search (bool): if True, use grid search to select the optimal ridge_factor
        basis_scale_factor (float): a multiplier to basis in order to avoid extremely small value
        dropout_model_type (str): model used for dropout model, select from "linear", "mlp" and "rf" (only applies to MAR case)
        dropout_scale_obs (bool): if True, scale observation before fitting the model
        model_suffix (str): suffix to the filename of saved model
        prob_lbound (float): lower bound of dropout/survival probability to avoid extreme inverse weight
        eval_env (gym.Env): dynamic environment to evaluate the policy, if not specified, use env
        filename_data (str): path to the observed data (csv file)
        filename_train (str): path to the training results

    Returns:
        if_converge (bool)
    """
    
    if export_dir:
        pathlib.Path(os.path.join(export_dir)).mkdir(parents=True, exist_ok=True)
    assert env is not None, 'Please specify the environment'
    env.T = T
    if env.is_vector_env:
        env.num_envs = n
    agent = LSTDQ(env=env,
                       n=n,
                       scale=scale,
                       product_tensor=product_tensor,
                       discount=discount,
                       eval_env=eval_env,
                       basis_scale_factor=basis_scale_factor)
    agent.dropout_obs_count_thres = max(dropout_obs_count_thres - 1,0) # -1 because this is the index
    # generate data and basis spline
    agent.gen_masked_buffer(#policy=agent.obs_policy,
                            S_inits=S_inits,
                            total_N=total_N,
                            burn_in=burn_in,
                            seed=seed)
    print(f'dropout rate: {agent.dropout_rate}')
    print(f'missing rate: {agent.missing_rate}')
    print(f'n: {agent.n}')
    print(f'total_N: {agent.total_N}')

    spline_degree = d
    if not L:
        L = max(4, int(((agent.total_N)**(3 / 7))**(1 / env.state_dim))) if knots is None else len(knots) - 1 - spline_degree

    config_str = f'configuraion: T = {T}, n = {n}, discount = {discount}, L = {L}, ipw = {ipw}, aipw = False, estimate_missing_prob = {estimate_missing_prob}'
    print(config_str)

    agent.B_spline(L=max(3, L), d=spline_degree, knots=knots)
    
    if missing_mechanism is None:
        # no missingness, hence no need of adjustment
        ipw = False
        estimate_missing_prob = False

    if estimate_missing_prob:
        print(f'Fit dropout model : {missing_mechanism}')
        pathlib.Path(os.path.join(export_dir, 'models')).mkdir(parents=True,
                                                               exist_ok=True)
        print('Start fitting dropout model...')
        fit_dropout_start = time.time()
        agent.train_dropout_model(
            model_type=dropout_model_type,
            dropout_model = dropout_model,
            missing_mechanism=missing_mechanism,
            train_ratio=0.8,
            scale_obs=dropout_scale_obs,
            dropout_obs_count_thres=dropout_obs_count_thres,
            export_dir=os.path.join(export_dir, 'models'),
            pkl_filename=
            f"dropout_model_{dropout_model_type}_T{T}_n{n}_gamma{discount}_{model_suffix}.pkl",
            seed=seed,
            include_reward=dropout_include_reward,
            instrument_var_index=instrument_var_index,
            mnar_y_transform=mnar_y_transform,
            psi_init=psi_init, 
            bandwidth_factor=bandwidth_factor,
            verbose=True)
        print('Start estimating dropout probability...')
        agent.estimate_missing_prob(missing_mechanism=missing_mechanism)
        fit_dropout_end = time.time()
        print(f'Finished! {fit_dropout_end-fit_dropout_start} secs elapsed.')

    # estimate beta
    print("start updating Q-function for target policy...")
    agent._beta_hat(policy=target_policy,
                    ipw=ipw,
                    estimate_missing_prob=estimate_missing_prob,
                    weight_curr_step=weight_curr_step,
                    prob_lbound=prob_lbound,
                    ridge_factor=ridge_factor,
                    grid_search=grid_search,
                    verbose=True,
                    subsample_index=None)

    if grid_search:
        print(agent._ridge_param_cv)

    print("end updating...")

    buffer_data = agent.export_buffer(eval_Q=True)
    # buffer_data.to_csv(os.path.join(export_dir, filename_data), index=False)
    
    # check missing rate
    stepwise_missing_rate = dict()
    stepwise_missing_rate[1] = 0
    total = buffer_data['id'].nunique()
    buffer_data = buffer_data.loc[~buffer_data['action'].isna()] # remove the terminal state
    count = Counter(buffer_data.groupby('id').size())
    for t in range(2, T + 1 - burn_in):
        stepwise_missing_rate[t] = sum([count[x] for x in range(1, t)]) / total
    print("stepwise missing rate:", stepwise_missing_rate)

    env.close()
    # pickle the trained model
    if export_dir:
        filename_train = os.path.join(export_dir, filename_train)
    with open(filename_train, 'wb') as outfile_train:
        pickle.dump(
            {
                'scaler':
                agent.scaler,
                'knot':
                agent.knot,
                'bspline':
                agent.bspline,
                'basis_scale_factor':
                agent.basis_scale_factor,
                'est_beta':
                agent.est_beta,
                'para':
                agent.para,
                'para_dim':
                agent.para_dim,
                'init_obs':
                agent._initial_obs,
                'Sigma_hat':
                agent.Sigma_hat,
                'vector':
                agent.vector,
                'inv_Sigma_hat':
                agent.inv_Sigma_hat,
                'max_inverse_wt':
                agent.max_inverse_wt,
                'stepwise_missing_rate': 
                stepwise_missing_rate,
            }, outfile_train)
    del agent
    _ = gc.collect()

def get_target_value_multi(T=25,
                          n=500,
                          env=None,
                          eval_T=250,
                          vf_mc_size=10000,
                          target_policy=None,
                          use_vector_env=False,
                          import_dir=None,
                          filename_train=None,
                          filename_true_value=None,
                          export_dir=None,
                          value_import_dir=None,
                          scale="NormCdf",
                          product_tensor=True,
                          discount=0.5,
                          eval_env=None,
                          dropout_model = 'semiparam',
                          filename_value='value',
                          eval_S_inits_dict=None,
                          eval_seed=None):
    """Get the value of target policy under multiple initial state distributions"""

    if export_dir:
        pathlib.Path(os.path.join(export_dir)).mkdir(parents=True, exist_ok=True)
    if import_dir:
        filename_train = os.path.join(import_dir, filename_train)
    with open(filename_train, 'rb') as outfile_train:
        output = pickle.load(outfile_train)
    assert env is not None, 'Please specify the environment'
    env.T = T
    if env.is_vector_env:
        env.num_envs = n
    b = LSTDQ(env=env,
                   n=n,
                   scale=scale,
                   product_tensor=product_tensor,
                   discount=discount,
                   eval_env=eval_env)
    b.scaler = output['scaler']
    b.bspline = output['bspline']
    b.para = output['para']
    b.para_dim = output['para_dim']
    b._initial_obs = output['init_obs']
    b.basis_scale_factor = output['basis_scale_factor']
    b.est_beta = output['est_beta']
    np.random.seed(seed=eval_seed)
    if eval_S_inits_dict is None:
        eval_S_inits_dict = {}
        eval_S_inits_dict['C'] = b.sample_initial_states(size=vf_mc_size,
                                          from_data=True,
                                          seed=eval_seed)
    value_store = {eval_T: {}}
    initial_scenario_list = list(eval_S_inits_dict.keys())
    t = eval_T
    b.eval_env.T = eval_T
    for k in eval_S_inits_dict.keys():
        value_store[eval_T][k] = defaultdict(list)
        if isinstance(eval_S_inits_dict[k], dict):
            value_store[eval_T][k]['initial_states'] = None
            value_store[eval_T][k]['initial_states_kwargs'] = eval_S_inits_dict[k]
        else:
            value_store[eval_T][k]['initial_states'] = eval_S_inits_dict[k]
            value_store[eval_T][k]['initial_states_kwargs'] = None
    if use_vector_env:
        action_levels = env.single_action_space.n
    else:
        action_levels = env.action_space.n
    eval_S_inits_sample = {}
    eval_S_inits_sample_kwargs = {}
    if value_import_dir and filename_true_value and os.path.exists(os.path.join(value_import_dir, filename_true_value)):
        with open(os.path.join(value_import_dir, filename_true_value),'rb') as outfile:
            true_value_dict = pickle.load(outfile)
        true_value = {}
        true_Q = {}
        for k in eval_S_inits_dict.keys():
            eval_S_inits_sample[k] = true_value_dict[eval_T][k].get('initial_states', None)
            eval_S_inits_sample_kwargs[k] = true_value_dict[eval_T][k].get('initial_states_kwargs', None)
            assert eval_S_inits_sample[k] is not None
            if isinstance(eval_S_inits_sample[k], list):
                eval_S_inits_sample[k] = np.array(eval_S_inits_sample[k])
            if not isinstance(eval_S_inits_dict[k], dict):
                assert (eval_S_inits_sample[k] == eval_S_inits_dict[k]).all() or \
                    np.mean((eval_S_inits_sample[k] - eval_S_inits_dict[k]) ** 2) < 1e-12 # tolerate small numerical difference
            if eval_S_inits_sample_kwargs[k] is not None and isinstance(eval_S_inits_dict[k], dict):
                for k1 in eval_S_inits_sample_kwargs[k]:
                    assert (eval_S_inits_sample_kwargs[k][k1] == eval_S_inits_dict[k][k1]).all()
                for k2 in eval_S_inits_dict[k]:
                    assert (eval_S_inits_sample_kwargs[k][k2] == eval_S_inits_dict[k][k2]).all()
            true_value[k] = true_value_dict[eval_T][k]['actual_value']
            true_Q[k] = {}
            for a in range(action_levels):
                true_Q[k][a] = true_value_dict[eval_T][k][f'actual_Q_{a}']
    else:
        true_value = {}
        true_Q = {}
        true_value_dict = {eval_T: {}}
        for k in eval_S_inits_dict.keys():
            print(f'Evaluate policy under initial scenario {k}...')
            if isinstance(eval_S_inits_dict[k], dict):
                eval_S_inits_sample[k], true_value[k], true_Q[k] = b.evaluate_pointwise_Q(
                    policy=target_policy,
                    seed=eval_seed,
                    S_inits=None,
                    S_inits_kwargs=eval_S_inits_dict[k],
                    eval_size=vf_mc_size,
                    eval_horizon=eval_T,
                    pointwise_eval_size=50
                )
                eval_S_inits_sample_kwargs[k] = eval_S_inits_dict[k]
            else:
                eval_S_inits_sample[k], true_value[k], true_Q[k] = b.evaluate_pointwise_Q(
                    policy=target_policy,
                    seed=eval_seed,
                    S_inits=eval_S_inits_dict[k],
                    S_inits_kwargs={},
                    eval_size=vf_mc_size,
                    eval_horizon=eval_T,
                    pointwise_eval_size=50
                )
                eval_S_inits_sample_kwargs[k] = {}
            if isinstance(eval_S_inits_sample[k], list):
                eval_S_inits_sample[k] = np.array(eval_S_inits_sample[k])
            true_value_dict[eval_T][k] = {}
            true_value_dict[eval_T][k]['initial_states'] = eval_S_inits_sample[k]
            true_value_dict[eval_T][k]['initial_states_kwargs'] = eval_S_inits_sample_kwargs[k]
            true_value_dict[eval_T][k]['actual_value'] = true_value[k]
            for a in range(action_levels):
                true_value_dict[eval_T][k][f'actual_Q_{a}'] = true_Q[k][a]
        with open(os.path.join(value_import_dir, filename_true_value),'wb') as outfile:
            pickle.dump(true_value_dict, outfile)
    if eval_S_inits_sample:
        eval_S_inits_dict = eval_S_inits_sample
    for k in eval_S_inits_dict.keys():
        value_store[t][k]['initial_states'] = eval_S_inits_sample[k]
        value_store[t][k]['initial_states_kwargs'] = eval_S_inits_sample_kwargs[k]
    if use_vector_env:
        for k in eval_S_inits_dict.keys():
            value_store[t][k]['actual_value'] = true_value[k]
            est_value = b.V(states=eval_S_inits_dict[k], policy=target_policy)
            value_store[t][k]['est_value'] = est_value.squeeze().tolist()
            for a in range(action_levels):
                value_store[t][k][f'actual_Q_{a}'] = true_Q[k][a]
                A_inits = np.repeat(a, repeats=len(eval_S_inits_dict[k]))
                est_Q = b.Q(states=eval_S_inits_dict[k], actions=A_inits)
                value_store[t][k][f'est_Q_{a}'] = est_Q.squeeze().tolist()
    else:
        for k in eval_S_inits_dict.keys():
            value_store[t][k]['actual_value'] = true_value[k]
            est_value = b.V(states=eval_S_inits_dict[k], policy=target_policy)
            value_store[t][k]['est_value'] = est_value.squeeze().tolist()
            for a in range(action_levels):
                value_store[t][k][f'actual_Q_{a}'] = true_Q[k][a]
                A_inits = np.repeat(a, repeats=len(eval_S_inits_dict[k]))
                est_Q = b.Q(states=eval_S_inits_dict[k], actions=A_inits)
                value_store[t][k][f'est_Q_{a}'] = est_Q.squeeze().tolist()

    for k in eval_S_inits_dict.keys():
        value_store[t][k]['actual_value_int'].append(
            np.mean(value_store[t][k]['actual_value']))
        value_store[t][k]['est_value_int'].append(
            np.mean(value_store[t][k]['est_value']))
        value_store[t][k]['MeanSE'] = np.mean(
            (np.array(value_store[t][k]['est_value']) -
                np.array(value_store[t][k]['actual_value']))**2)
        value_store[t][k]['MaxSE'] = np.max(
            (np.array(value_store[t][k]['est_value']) -
                np.array(value_store[t][k]['actual_value']))**2)
    env.close()
    if export_dir:
        filename_value = os.path.join(export_dir, filename_value)
    with open(filename_value, 'wb') as outfile:
        pickle.dump(value_store, outfile)
    return value_store

def eval_V_int_CI_multi(
        T=30,
        n=500,
        env=None,
        L=None,
        d=3,
        eval_T=250,
        discount=0.8,
        knots=None,
        total_N=None,
        burn_in=0,
        ipw=False,
        weight_curr_step=True,
        estimate_missing_prob=False,
        prob_lbound=1e-3,
        ridge_factor=1e-9,
        train_S_inits=None, 
        train_S_inits_kwargs={},
        basis_scale_factor=1.,
        vf_mc_size=10000,
        mc_size=50,
        alpha_list = [0.05],
        target_policy=None,
        dropout_model_type='linear',
        dropout_model = 'semiparam',
        dropout_obs_count_thres=1,
        dropout_scale_obs=False,
        dropout_include_reward=False,
        missing_mechanism='mar',
        instrument_var_index=None,
        mnar_y_transform=None,
        psi_init=None,
        bandwidth_factor=1.5,
        value_import_dir=None,
        export_dir=None,
        filename_true_value=None,
        filename_est_value=None,
        filename_CI=None,
        figure_name='est_value_scatterplot.png',
        result_filename=None,
        scale="NormCdf",
        product_tensor=True,
        eval_env=None,
        eval_S_inits_dict=None,
        eval_seed=1,
        verbose_freq=1):
    """Calculate the CI for integrated value for multiple initial state distribution."""

    if value_import_dir:
        pathlib.Path(value_import_dir).mkdir(parents=True, exist_ok=True)
    if not result_filename:
        result_filename = "result_T_%d_n_%d_S_integration_L_%d.txt" % (T, n, L)
    if export_dir:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
        filename_CI = os.path.join(export_dir, filename_CI)
        result_filename = os.path.join(export_dir, result_filename)
    if export_dir:
        est_value_filename = os.path.join(export_dir, filename_est_value)
    else:
        est_value_filename = filename_est_value
    # make sure alpha is a list of significance levels
    if not hasattr(alpha_list, '__iter__'):
        alpha_list = list(alpha_list)
    if missing_mechanism is None:
        # no missingness, hence no need of adjustment
        ipw = False
        estimate_missing_prob = False

    assert env is not None, 'Please specify the environment'
    env.T = T
    if env.is_vector_env:
        env.num_envs = n
    agent = LSTDQ(env=env,
                       n=n,
                       scale=scale,
                       product_tensor=product_tensor,
                       discount=discount,
                       eval_env=eval_env,
                       basis_scale_factor=basis_scale_factor)
    agent.dropout_obs_count_thres = max(dropout_obs_count_thres-1,0) # -1 because this is the index

    spline_degree = d
    if not L:
        L = int(np.sqrt(
            (agent.total_N)
            **(3 / 7))) if knots is None else len(knots) - 1 - spline_degree

    np.random.seed(seed=eval_seed)
    if eval_S_inits_dict is None:
        eval_S_inits_dict = {}
        eval_S_inits_dict['C'] = agent.sample_initial_states(size=vf_mc_size,
                                          from_data=True,
                                          seed=eval_seed)
    eval_S_inits_sample = {}
    eval_S_inits_sample_kwargs = {}
    if value_import_dir and filename_true_value and os.path.exists(os.path.join(value_import_dir, filename_true_value)):
        with open(os.path.join(value_import_dir, filename_true_value),
                  'rb') as outfile:
            true_value_dict = pickle.load(outfile)
        est_mean = {}
        true_value = {}
        for k in eval_S_inits_dict.keys():
            eval_S_inits_sample[k] = true_value_dict[k].get('initial_states', None)
            eval_S_inits_sample_kwargs[k] = true_value_dict[k].get('initial_states_kwargs', None)
            assert eval_S_inits_sample[k] is not None
            if isinstance(eval_S_inits_sample[k], list):
                eval_S_inits_sample[k] = np.array(eval_S_inits_sample[k])
            if not isinstance(eval_S_inits_dict[k], dict):
                print(f'eval_S_inits_sample[k]: {eval_S_inits_sample[k]}')
                print(f'eval_S_inits_dict[k]: {eval_S_inits_dict[k]}')
                assert (eval_S_inits_sample[k].astype('float32') == eval_S_inits_dict[k].astype('float32')).all()
            if eval_S_inits_sample_kwargs[k] is not None and isinstance(eval_S_inits_dict[k], dict):
                print(f'eval_S_inits_sample_kwargs[k]:{eval_S_inits_sample_kwargs[k]}')
                print(f'eval_S_inits_dict[k]:{eval_S_inits_dict[k]}')
                for k1 in eval_S_inits_sample_kwargs[k]:
                    assert (eval_S_inits_sample_kwargs[k][k1] == eval_S_inits_dict[k][k1]).all()
                for k2 in eval_S_inits_dict[k]:
                    assert (eval_S_inits_sample_kwargs[k][k2] == eval_S_inits_dict[k][k2]).all()
            est_mean[k] = true_value_dict[k]['true_V_int']
            true_value[k] = true_value_dict[k]['true_V_list']
        print(f'true integrated V: {est_mean}')
    else:
        est_mean = {}
        true_value = {}
        true_Q = {}
        true_value_dict = {}
        for k in eval_S_inits_dict.keys():
            print(f'Evaluate policy under initial scenario {k}...')
            if isinstance(eval_S_inits_dict[k], dict):
                eval_S_inits_sample[k], true_value[k], true_Q[k] = agent.evaluate_pointwise_Q(
                    policy=target_policy,
                    seed=eval_seed,
                    S_inits=None,
                    S_inits_kwargs=eval_S_inits_dict[k],
                    eval_size=vf_mc_size,
                    eval_horizon=eval_T,
                    pointwise_eval_size=50
                )
                eval_S_inits_sample_kwargs[k] = eval_S_inits_dict[k]
            else:
                eval_S_inits_sample[k], true_value[k], true_Q[k] = agent.evaluate_pointwise_Q(
                    policy=target_policy,
                    seed=eval_seed,
                    S_inits=eval_S_inits_dict[k],
                    S_inits_kwargs={},
                    eval_size=vf_mc_size,
                    eval_horizon=eval_T,
                    pointwise_eval_size=50
                )
                eval_S_inits_sample_kwargs[k] = {}
            if isinstance(eval_S_inits_sample[k], list):
                eval_S_inits_sample[k] = np.array(eval_S_inits_sample[k])
            est_mean[k] = np.mean(true_value[k])
            true_value_dict[k] = {
                'initial_states': eval_S_inits_sample[k],
                'initial_states_kwargs': eval_S_inits_sample_kwargs[k],
                'true_V_int': est_mean[k], 
                'true_V_list': true_value[k]
                }
        print(f'true integrated V: {est_mean}')
        with open(os.path.join(value_import_dir, filename_true_value),'wb') as outfile:
            pickle.dump(true_value_dict, outfile)
    if eval_S_inits_sample:
        eval_S_inits_dict = eval_S_inits_sample

    est_V_int_list = defaultdict(list)
    est_V_int_std_list = defaultdict(list)
    est_V_mse_list = defaultdict(list)
    max_inverse_wt_list = []
    mnar_psi_est_list, dropout_prob_mse_list = [], []
    beta_list, Sigma_hat_list, inv_Sigma_hat_list, vector_list = [], [], [], []
    i = 0
    count = {k: defaultdict(int) for k in eval_S_inits_dict.keys()}
    lengths = {k: defaultdict(list) for k in eval_S_inits_dict.keys()}
    intervals = {k: defaultdict(list) for k in eval_S_inits_dict.keys()}
    while i < mc_size:
        seed = i
        np.random.seed(seed)
        agent = LSTDQ(env=env,
                        n=n,
                        scale=scale,
                        product_tensor=product_tensor,
                        discount=discount,
                        eval_env=eval_env,
                        basis_scale_factor=basis_scale_factor)
        agent.dropout_obs_count_thres = max(dropout_obs_count_thres-1,0) # -1 because this is the index
        agent.masked_buffer = {} # attention: when using gen_masked_buffer, we should empty the buffer first!
        agent.gen_masked_buffer(#policy=agent.obs_policy,
                                S_inits=train_S_inits,
                                S_inits_kwargs=train_S_inits_kwargs,
                                total_N=total_N,
                                burn_in=burn_in,
                                seed=seed)
        print(f'dropout rate: {agent.dropout_rate}')
        print(f'missing rate: {agent.missing_rate}')
        print(f'total_N: {agent.total_N}')
        agent.B_spline(L=max(3, L), d=spline_degree, knots=knots)

        if estimate_missing_prob:
            print(f'Fit dropout model : {missing_mechanism}')
            pathlib.Path(os.path.join(export_dir,
                                      'models')).mkdir(parents=True,
                                                       exist_ok=True)
            print('Start fitting dropout model...')
            fit_dropout_start = time.time()
            agent.train_dropout_model(
                model_type=dropout_model_type,
                dropout_model = dropout_model,
                missing_mechanism=missing_mechanism,
                train_ratio=0.8,
                scale_obs=dropout_scale_obs,
                dropout_obs_count_thres=dropout_obs_count_thres,
                export_dir=os.path.join(export_dir, 'models'),
                pkl_filename=
                f"dropout_model_{dropout_model_type}_T{T}_n{n}_gamma{discount}_itr_{i}.pkl",
                seed=seed,
                include_reward=dropout_include_reward, # True, False
                instrument_var_index=instrument_var_index,
                mnar_y_transform=mnar_y_transform,
                psi_init=psi_init,
                bandwidth_factor=bandwidth_factor,
                verbose=True)
            agent.estimate_missing_prob(missing_mechanism=missing_mechanism)
            fit_dropout_end = time.time()
            print(f'Finished! {fit_dropout_end-fit_dropout_start} secs elapsed.')
            dropout_prob_mse_list.append(agent._dropout_prob_mse)
            if missing_mechanism.lower() == 'mnar':
                mnar_psi_est_list.append(agent.fitted_dropout_model['model'].psi_hat)

        print('Make inference on the integrated value...')
        for k in eval_S_inits_dict.keys():
            eval_S_inits = eval_S_inits_dict[k]
            for sig_level in alpha_list:
                inference_dict = agent.inference_int(
                    policy=target_policy,
                    alpha=sig_level,
                    ipw=ipw,
                    estimate_missing_prob=estimate_missing_prob,
                    weight_curr_step=weight_curr_step,
                    prob_lbound=prob_lbound,
                    ridge_factor=ridge_factor,
                    MC_size=vf_mc_size,
                    S_inits=eval_S_inits,
                    verbose=True)
                intervals[k][sig_level].append((inference_dict['lower_bound'], inference_dict['upper_bound']))
                lengths[k][sig_level].append(inference_dict['upper_bound'] - inference_dict['lower_bound'])
                if inference_dict['lower_bound'] < est_mean[k] < inference_dict['upper_bound']:
                    count[k][sig_level] += 1
                if i % verbose_freq == 0:
                    print(f"iteration {i}, inital_scenario {k}, alpha {sig_level}, count {count[k]}, CI ({inference_dict['lower_bound']},{inference_dict['upper_bound']}), ", 
                    f"sigma {(agent.V_int_sigma_sq/agent.total_T_ipw)**0.5}, ", 
                    f"estimated mean {np.mean([inference_dict['lower_bound'], inference_dict['upper_bound']])}, true mean {est_mean[k]}\n")
            
            est_V = inference_dict['value']
            est_V_std = inference_dict['std']
            pointwise_V_est = agent.V(states=eval_S_inits, policy=target_policy).squeeze()
            pointwise_mse = np.mean((pointwise_V_est - true_value[k]) ** 2)
            abnormal_itr = False
            if abnormal_itr:
                continue
            est_V_int_list[k].append(est_V)
            est_V_int_std_list[k].append(est_V_std)
            est_V_mse_list[k].append(pointwise_mse)
        beta = np.concatenate(list(agent.para.values())).reshape(-1)
        Sigma_hat = agent.Sigma_hat
        inv_Sigma_hat = agent.inv_Sigma_hat
        vector = agent.vector.reshape(-1)
        if not abnormal_itr:
            max_inverse_wt_list.append(agent.max_inverse_wt)
            beta_list.append(beta)
            Sigma_hat_list.append(Sigma_hat)
            inv_Sigma_hat_list.append(inv_Sigma_hat)
            vector_list.append(vector)
            i += 1
        del agent
        _ = gc.collect()
        with open(filename_CI, 'wb') as outfile_CI:
            pickle.dump(intervals, outfile_CI)
        output_dict = {
                'initial_states': eval_S_inits_sample,
                'initial_states_kwargs': eval_S_inits_sample_kwargs,
                'est_V_int_list': est_V_int_list,
                'est_V_int_std_list': est_V_int_std_list,
                'true_V_int': est_mean,
                'est_V_mse_list': est_V_mse_list,
                'max_inverse_wt_list': max_inverse_wt_list,
                'mnar_psi_est_list': mnar_psi_est_list if ipw and estimate_missing_prob and missing_mechanism.lower() == 'mnar' else None,
                'dropout_prob_mse_list': dropout_prob_mse_list if ipw else None,
                'beta_list': beta_list,
                'Sigma_hat_list': Sigma_hat_list,
                'inv_Sigma_hat_list': inv_Sigma_hat_list,
                'vector_list': vector_list
            }
        with open(est_value_filename, "wb") as f:
            pickle.dump(output_dict, f)
    for k in eval_S_inits_dict.keys():
        for sig_level in alpha_list:
            print(f'initial scenario {k}, alpha {sig_level}, coverage prob: {count[k][sig_level] / len(lengths[k][sig_level])}')
            result_filename_suffix = result_filename.rstrip('.txt') + f'_init_{k}_alpha{sig_level}.txt'
            with open(result_filename_suffix, "a+") as f:
                f.write("Count %d in %d, ratio %f \n" %
                        (count[k][sig_level], len(lengths[k][sig_level]), count[k][sig_level] / len(lengths[k][sig_level])))
                f.write("Average lengths %f \n" % np.mean(lengths[k][sig_level]))

def eval_V_int_CI_bootstrap_multi(
        T=30,
        n=500,
        env=None,
        L=None,
        d=3,
        eval_T=250,
        discount=0.8,
        knots=None,
        total_N=None,
        burn_in=0,
        ipw=False,
        weight_curr_step=True,
        estimate_missing_prob=False,
        prob_lbound=1e-3,
        ridge_factor=1e-9,
        train_S_inits=None, 
        train_S_inits_kwargs={},
        basis_scale_factor=1.,
        vf_mc_size=10000,
        mc_size=50,
        alpha_list = [0.05],
        bootstrap_size=100,
        target_policy=None,
        dropout_model_type='linear',
        dropout_model = 'semiparam',
        dropout_obs_count_thres=1, # 0
        dropout_scale_obs=False,
        dropout_include_reward=False,
        missing_mechanism='mar',
        instrument_var_index=None,
        mnar_y_transform=None,
        psi_init=None,
        bandwidth_factor=1.5,
        value_import_dir=None,
        export_dir=None,
        filename_true_value=None,
        filename_est_value=None,
        filename_CI=None,
        filename_bootstrap_CI=None,
        figure_name='est_value_scatterplot.png',
        result_filename=None,
        scale="NormCdf",
        product_tensor=True,
        eval_env=None,
        eval_S_inits_dict=None,
        eval_seed=1,
        verbose_freq=1):
    """Calculate the CI for integrated value for multiple initial state distribution, user bootstrap to estimate standard deviation."""

    if value_import_dir:
        pathlib.Path(value_import_dir).mkdir(parents=True, exist_ok=True)
    if not result_filename:
        result_filename = "result_bootstrap_T_%d_n_%d_S_integration_L_%d.txt" % (T, n, L)
    if export_dir:
        pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
        filename_CI = os.path.join(export_dir, filename_CI)
        filename_bootstrap_CI = os.path.join(export_dir, filename_bootstrap_CI)
        result_filename = os.path.join(export_dir, result_filename)
    if export_dir:
        est_value_filename = os.path.join(export_dir, filename_est_value)
    else:
        est_value_filename = filename_est_value
    # make sure alpha is a list of significance levels
    if not hasattr(alpha_list, '__iter__'):
        alpha_list = list(alpha_list)
    if missing_mechanism is None:
        # no missingness, hence no need of adjustment
        ipw = False
        estimate_missing_prob = False

    assert env is not None, 'Please specify the environment'
    env.T = T
    if env.is_vector_env:
        env.num_envs = n
    agent = LSTDQ(env=env,
                       n=n,
                       scale=scale,
                       product_tensor=product_tensor,
                       discount=discount,
                       eval_env=eval_env,
                       basis_scale_factor=basis_scale_factor)
    agent.dropout_obs_count_thres = max(dropout_obs_count_thres-1,0) # -1 because this is the index

    spline_degree = d
    if not L:
        L = int(np.sqrt(
            (agent.total_N)
            **(3 / 7))) if knots is None else len(knots) - 1 - spline_degree
    
    np.random.seed(seed=eval_seed)
    if eval_S_inits_dict is None:
        eval_S_inits_dict = {}
        eval_S_inits_dict['C'] = agent.sample_initial_states(size=vf_mc_size,
                                          from_data=True,
                                          seed=eval_seed)
    eval_S_inits_sample = {}
    eval_S_inits_sample_kwargs = {}
    if value_import_dir and filename_true_value and os.path.exists(os.path.join(value_import_dir, filename_true_value)):
        with open(os.path.join(value_import_dir, filename_true_value),
                  'rb') as outfile:
            true_value_dict = pickle.load(outfile)
        est_mean = {}
        true_value = {}
        for k in eval_S_inits_dict.keys():
            eval_S_inits_sample[k] = true_value_dict[k].get('initial_states', None)
            eval_S_inits_sample_kwargs[k] = true_value_dict[k].get('initial_states_kwargs', None)
            assert eval_S_inits_sample[k] is not None
            if isinstance(eval_S_inits_sample[k], list):
                eval_S_inits_sample[k] = np.array(eval_S_inits_sample[k])
            if not isinstance(eval_S_inits_dict[k], dict):
                assert (eval_S_inits_sample[k].astype('float32') == eval_S_inits_dict[k].astype('float32')).all()
            if eval_S_inits_sample_kwargs[k] is not None and isinstance(eval_S_inits_dict[k], dict):
                print(f'eval_S_inits_sample_kwargs[k]:{eval_S_inits_sample_kwargs[k]}')
                print(f'eval_S_inits_dict[k]:{eval_S_inits_dict[k]}')
                for k1 in eval_S_inits_sample_kwargs[k]:
                    assert (eval_S_inits_sample_kwargs[k][k1] == eval_S_inits_dict[k][k1]).all()
                for k2 in eval_S_inits_dict[k]:
                    assert (eval_S_inits_sample_kwargs[k][k2] == eval_S_inits_dict[k][k2]).all()
            est_mean[k] = true_value_dict[k]['true_V_int']
            true_value[k] = true_value_dict[k]['true_V_list']
        print(f'true integrated V: {est_mean}')
    else:
        est_mean = {}
        true_value = {}
        true_Q = {}
        true_value_dict = {}
        for k in eval_S_inits_dict.keys():
            print(f'Evaluate policy under initial scenario {k}...')
            if isinstance(eval_S_inits_dict[k], dict):
                eval_S_inits_sample[k], true_value[k], true_Q[k] = agent.evaluate_pointwise_Q(
                    policy=target_policy,
                    seed=eval_seed,
                    S_inits=None,
                    S_inits_kwargs=eval_S_inits_dict[k],
                    eval_size=vf_mc_size,
                    eval_horizon=eval_T,
                    pointwise_eval_size=50)
                eval_S_inits_sample_kwargs[k] = eval_S_inits_dict[k]
            else:
                eval_S_inits_sample[k], true_value[k], true_Q[k] = agent.evaluate_pointwise_Q(
                    policy=target_policy,
                    seed=eval_seed,
                    S_inits=eval_S_inits_dict[k],
                    S_inits_kwargs={},
                    eval_size=vf_mc_size,
                    eval_horizon=eval_T,
                    pointwise_eval_size=50
                )
                eval_S_inits_sample_kwargs[k] = {}       
            if isinstance(eval_S_inits_sample[k], list):
                eval_S_inits_sample[k] = np.array(eval_S_inits_sample[k])         
            est_mean[k] = np.mean(true_value[k])
            true_value_dict[k] = {
                'initial_states': eval_S_inits_sample[k],
                'initial_states_kwargs': eval_S_inits_sample_kwargs[k],
                'true_V_int': est_mean[k], 
                'true_V_list': true_value[k]
            }
        print(f'true integrated V: {est_mean}')
        with open(os.path.join(value_import_dir, filename_true_value),'wb') as outfile:
            pickle.dump(true_value_dict, outfile)
    if eval_S_inits_sample:
        eval_S_inits_dict = eval_S_inits_sample

    est_V_int_list = defaultdict(list)
    est_V_int_std_list = defaultdict(list)
    bootstrap_V_int_std_list = defaultdict(list)
    est_V_mse_list = defaultdict(list)
    max_inverse_wt_list = []
    mnar_psi_est_list, dropout_prob_mse_list = [], []
    beta_list, Sigma_hat_list, inv_Sigma_hat_list, vector_list = [], [], [], []
    i = 0
    count = {k: defaultdict(int) for k in eval_S_inits_dict.keys()}
    lengths = {k: defaultdict(list) for k in eval_S_inits_dict.keys()}
    intervals = {k: defaultdict(list) for k in eval_S_inits_dict.keys()}
    bootstrap_count = {k: defaultdict(int) for k in eval_S_inits_dict.keys()}
    bootstrap_lengths = {k: defaultdict(list) for k in eval_S_inits_dict.keys()}
    bootstrap_intervals = {k: defaultdict(list) for k in eval_S_inits_dict.keys()}

    while i < mc_size:
        seed = i
        np.random.seed(seed)
        agent = LSTDQ(env=env,
                        n=n,
                        scale=scale,
                        product_tensor=product_tensor,
                        discount=discount,
                        eval_env=eval_env,
                        basis_scale_factor=basis_scale_factor)
        agent.dropout_obs_count_thres = max(dropout_obs_count_thres-1,0) # dropout_obs_count_thres
        agent.masked_buffer = {} # attention: when using gen_masked_buffer, we should empty the buffer first!
        agent.gen_masked_buffer(#policy=agent.obs_policy,
                                S_inits=train_S_inits,
                                S_inits_kwargs=train_S_inits_kwargs,
                                total_N=total_N,
                                burn_in=burn_in,
                                seed=seed)
        print(f'dropout rate: {agent.dropout_rate}')
        print(f'missing rate: {agent.missing_rate}')
        print(f'total_N: {agent.total_N}')

        agent.B_spline(L=max(3, L), d=spline_degree, knots=knots)
        
        if estimate_missing_prob:
            print(f'Fit dropout model : {missing_mechanism}')
            pathlib.Path(os.path.join(export_dir,
                                      'models')).mkdir(parents=True,
                                                       exist_ok=True)
            print('Start fitting dropout model...')
            fit_dropout_start = time.time()
            agent.train_dropout_model(
                model_type=dropout_model_type,
                dropout_model = dropout_model,
                missing_mechanism=missing_mechanism,
                train_ratio=0.8,
                scale_obs=dropout_scale_obs,
                dropout_obs_count_thres=dropout_obs_count_thres,
                export_dir=os.path.join(export_dir, 'models'),
                pkl_filename=
                f"dropout_model_{dropout_model_type}_T{T}_n{n}_gamma{discount}_itr_{i}.pkl",
                seed=seed,
                include_reward=dropout_include_reward,
                instrument_var_index=instrument_var_index,
                mnar_y_transform=mnar_y_transform,
                psi_init=psi_init,
                bandwidth_factor=bandwidth_factor,
                verbose=True)
            agent.estimate_missing_prob(missing_mechanism=missing_mechanism)
            fit_dropout_end = time.time()
            print(f'Finished! {fit_dropout_end-fit_dropout_start} secs elapsed.')
            dropout_prob_mse_list.append(agent._dropout_prob_mse)
            if missing_mechanism.lower() == 'mnar':
                mnar_psi_est_list.append(agent.fitted_dropout_model['model'].psi_hat)

        print('Make inference on the integrated value...')
        for k in eval_S_inits_dict.keys():
            eval_S_inits = eval_S_inits_dict[k]
            for sig_level in alpha_list:
                inference_dict = agent.inference_int(
                    policy=target_policy,
                    alpha=sig_level,
                    ipw=ipw,
                    estimate_missing_prob=estimate_missing_prob,
                    weight_curr_step=weight_curr_step,
                    prob_lbound=prob_lbound,
                    ridge_factor=ridge_factor,
                    MC_size=vf_mc_size,
                    S_inits=eval_S_inits,
                    verbose=True)
                intervals[k][sig_level].append((inference_dict['lower_bound'], inference_dict['upper_bound']))
                lengths[k][sig_level].append(inference_dict['upper_bound'] - inference_dict['lower_bound'])
                if inference_dict['lower_bound'] < est_mean[k] < inference_dict['upper_bound']:
                    count[k][sig_level] += 1
            
            est_V = inference_dict['value']
            est_V_std = inference_dict['std']
            print(f'init {k} theoretical std: {est_V_std}')
            pointwise_V_est = agent.V(states=eval_S_inits, policy=target_policy).squeeze()
            pointwise_mse = np.mean((pointwise_V_est - true_value[k]) ** 2)
            abnormal_itr = False
            if abnormal_itr:
                continue
            est_V_int_list[k].append(est_V)
            est_V_int_std_list[k].append(est_V_std)
            est_V_mse_list[k].append(pointwise_mse)
        beta = np.concatenate(list(agent.para.values())).reshape(-1)
        Sigma_hat = agent.Sigma_hat
        inv_Sigma_hat = agent.inv_Sigma_hat
        vector = agent.vector.reshape(-1)
        max_inverse_wt_list.append(agent.max_inverse_wt)
        beta_list.append(beta)
        Sigma_hat_list.append(Sigma_hat)
        inv_Sigma_hat_list.append(inv_Sigma_hat)
        vector_list.append(vector)
            
        # run bootstrap
        buffer_key = list(agent.masked_buffer.keys())
        bootstrap_V_int_list = defaultdict(list)
        print('Start bootstrapping...')
        for j in range(bootstrap_size):
            selected_key = np.random.choice(buffer_key, size=len(buffer_key), replace=True)
            if estimate_missing_prob:
                agent.train_dropout_model(
                    model_type=dropout_model_type,
                    dropout_model = dropout_model,
                    missing_mechanism=missing_mechanism,
                    train_ratio=0.8,
                    scale_obs=dropout_scale_obs,
                    dropout_obs_count_thres=dropout_obs_count_thres,
                    subsample_index=selected_key,
                    export_dir=os.path.join(export_dir, 'models'),
                    pkl_filename=None,
                    seed=seed,
                    include_reward=dropout_include_reward,
                    instrument_var_index=instrument_var_index,
                    mnar_y_transform=mnar_y_transform,
                    psi_init=psi_init,
                    bandwidth_factor=bandwidth_factor,
                    verbose=False)
                agent.estimate_missing_prob(missing_mechanism=missing_mechanism,subsample_index=selected_key)
            agent._beta_hat(policy=target_policy,
                ipw=ipw,
                estimate_missing_prob=estimate_missing_prob,
                weight_curr_step=weight_curr_step,
                prob_lbound=prob_lbound,
                ridge_factor=ridge_factor,
                grid_search=False,
                verbose=False,
                subsample_index=selected_key)

            for k in eval_S_inits_dict.keys():
                eval_S_inits = eval_S_inits_dict[k]
                bootstrap_V_int = agent.V_int(policy=target_policy, MC_size=len(eval_S_inits), S_inits=eval_S_inits)
                bootstrap_V_int_list[k].append(bootstrap_V_int)
        for k in eval_S_inits_dict.keys():
            bootstrap_V_int_std = np.std(bootstrap_V_int_list[k], ddof=1)
            bootstrap_V_int_std_list[k].append(bootstrap_V_int_std)
            print(f'init {k} bootstrap std: {bootstrap_V_int_std}')
            for sig_level in alpha_list:
                bootstrap_lower_bound = np.quantile(a=bootstrap_V_int_list[k], q=sig_level/2)
                bootstrap_upper_bound = np.quantile(a=bootstrap_V_int_list[k], q=1-sig_level/2)
                bootstrap_intervals[k][sig_level].append((bootstrap_lower_bound, bootstrap_upper_bound))
                bootstrap_lengths[k][sig_level].append(bootstrap_upper_bound - bootstrap_lower_bound)
                if bootstrap_lower_bound < est_mean[k] < bootstrap_upper_bound:
                    bootstrap_count[k][sig_level] += 1
        print('Finished bootstrapping!')

        for k in eval_S_inits_dict.keys():
            if i % verbose_freq == 0:
                for sig_level in alpha_list:
                    print(f"iteration {i}, inital_scenario {k}, alpha {sig_level}, count {count[k]}, CI ({inference_dict['lower_bound']},{inference_dict['upper_bound']}), ", 
                    f"sigma {(agent.V_int_sigma_sq/agent.total_T_ipw)**0.5}, ", 
                    f"estimated mean {np.mean([inference_dict['lower_bound'], inference_dict['upper_bound']])}, true mean {est_mean[k]}\n")
        i += 1
        del agent
        _ = gc.collect()
        with open(filename_CI, 'wb') as outfile_CI:
            pickle.dump(intervals, outfile_CI)
        with open(filename_bootstrap_CI, 'wb') as outfile_CI:
            pickle.dump(bootstrap_intervals, outfile_CI)
        output_dict = {
                'initial_states': eval_S_inits_sample,
                'initial_states_kwargs': eval_S_inits_sample_kwargs,
                'est_V_int_list': est_V_int_list,
                'est_V_int_std_list': est_V_int_std_list,
                'bootstrap_V_int_std_list': bootstrap_V_int_std_list,
                'true_V_int': est_mean,
                'est_V_mse_list': est_V_mse_list,
                'max_inverse_wt_list': max_inverse_wt_list,
                'mnar_psi_est_list': mnar_psi_est_list if ipw and estimate_missing_prob and missing_mechanism.lower() == 'mnar' else None,
                'dropout_prob_mse_list': dropout_prob_mse_list if ipw else None,
                'beta_list': beta_list,
                'Sigma_hat_list': Sigma_hat_list,
                'inv_Sigma_hat_list': inv_Sigma_hat_list,
                'vector_list': vector_list
            }
        with open(est_value_filename, "wb") as f:
            pickle.dump(output_dict, f)
    for k in eval_S_inits_dict.keys():
        for sig_level in alpha_list:
            print(f'initial scenario {k}, alpha {sig_level}, coverage prob: {count[k][sig_level] / len(lengths[k][sig_level])}') # print(count / mc_size)
            result_filename_suffix = result_filename.rstrip('.txt') + f'_init_{k}_alpha{sig_level}.txt'
            with open(result_filename_suffix, "a+") as f:
                f.write("Count %d in %d, ratio %f \n" %
                        (count[k][sig_level], len(lengths[k][sig_level]), count[k][sig_level] / len(lengths[k][sig_level])))
                f.write("Average lengths %f \n" % np.mean(lengths[k][sig_level]))
                f.write("(bootstrap) Count %d in %d, ratio %f \n" %
                        (bootstrap_count[k][sig_level], len(bootstrap_lengths[k][sig_level]), bootstrap_count[k][sig_level] / len(bootstrap_lengths[k][sig_level])))
                f.write("(bootstrap) Average lengths %f \n" % np.mean(bootstrap_lengths[k][sig_level]))

