"""
Examine the coverage of CI for the integrated value.
"""

import os
import numpy as np
import pickle
from numpy.linalg import inv
from scipy.stats import norm, truncnorm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from custom_env.linear2d import Linear2dVectorEnv
    from ope_mnar.main import eval_V_int_CI_multi, eval_V_int_CI_bootstrap_multi
except:
    import sys
    sys.path.append(os.path.expanduser('~/ope_mnar/ope_mnar'))
    sys.path.append(os.path.expanduser('~/ope_mnar'))
    from custom_env.linear2d import Linear2dVectorEnv
    from ope_mnar.main import eval_V_int_CI_multi, eval_V_int_CI_bootstrap_multi

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str,
                    default='linear2d')
parser.add_argument('--max_episode_length', type=int, default=25)
parser.add_argument('--discount', type=float, default=0.8)
parser.add_argument('--num_trajs', type=int, default=500) # 250, 500
parser.add_argument('--burn_in', type=int, default=0)
parser.add_argument('--mc_size', type=int, default=250)
parser.add_argument('--eval_policy_mc_size', type=int,
                    default=100)  # use 100 for test purpose # default value = 10000
parser.add_argument('--eval_horizon', type=int, default=250)
parser.add_argument('--dropout_scheme', type=str, default='mnar.v0', choices=["0", "mnar.v0", "mar.v0"]) 
parser.add_argument('--dropout_rate', type=float, default=0.9)
parser.add_argument('--dropout_model', type=str, default='semiparam') # can take either 'param' or 'semiparam' method
parser.add_argument('--dropout_obs_count_thres', type=int, default=2)
parser.add_argument('--ipw',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--estimate_missing_prob',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--bootstrap',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=False)
parser.add_argument('--vectorize_env',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--log_suffix', type=str, default='')  
args = parser.parse_args()

if __name__ == '__main__':
    log_dir = os.path.expanduser('~/ope_mnar/output')
    env_class = args.env
    default_scaler = "MinMax"  # "NormCdf", "MinMax"

    T = args.max_episode_length
    n = args.num_trajs
    total_N = None  # T*n, None
    adaptive_dof = False
    dropout_rate = args.dropout_rate  # 0.9
    dropout_model = args.dropout_model
    gamma = args.discount
    mc_size = args.mc_size
    burn_in = args.burn_in
    # alpha = args.alpha
    alpha_list = [0.05] # [0.05, 0.1, 0.2, 0.3, 0.4]  # [0.05] # 
    vectorize_env = args.vectorize_env
    dropout_scheme = args.dropout_scheme
    dropout_obs_count_thres = args.dropout_obs_count_thres
    ipw = args.ipw
    estimate_missing_prob = args.estimate_missing_prob
    weight_curr_step = True
    instrument_var_index = None
    mnar_y_transform = None
    bandwidth_factor = None
    if env_class.lower() == 'linear2d':
        if dropout_scheme == '0':
            missing_mechanism = None
        elif dropout_scheme in ['mnar.v0', 'mnar.v1']:
            missing_mechanism = 'mnar'
            instrument_var_index = 1
            if dropout_scheme == 'mnar.v0':
                bandwidth_factor = 7.5
            elif dropout_scheme == 'mnar.v1':
                bandwidth_factor = 2.5
        else:
            missing_mechanism = 'mar'
    if missing_mechanism is None:
        # no missingness, hence no need of adjustment
        ipw = False
        estimate_missing_prob = False

    log_suffix = args.log_suffix
    prob_lbound = 1e-2  # 1e-2, 1e-3
    psi_true = None  # 1.5
    if missing_mechanism and missing_mechanism.lower() == 'mnar':
        plugin_psi_true = False
        initialize_with_psiT = True
    if not ipw:
        estimator = 'cc'
    elif not estimate_missing_prob:
        estimator = 'ipw_propT'
    else:
        estimator = 'ipw_propF'

    eval_policy_mc_size = args.eval_policy_mc_size  # 10000
    eval_horizon = args.eval_horizon  # 500
    initial_scenario_list = ['C']
    scale = default_scaler
    eval_seed = 123
    product_tensor = True
    spline_degree = 3
    if env_class == 'linear2d':
        if adaptive_dof:
            dof = max(4, int(((n * T)**(3/7))**(1/2)))  # degree of freedom
        else:
            dof = 7
    elif env_class == 'cartpole':
        if adaptive_dof:
            dof = max(4, int(((n * T)**(3/7))**(1/4))) # degree of freedom
        else:
            dof = 4
    ridge_factor = 1e-3  # 1e-9
    folder_suffix = '' # '_unscaled'
    folder_suffix += f'_missing{dropout_rate}'
    folder_suffix += f'_ridge{ridge_factor}'
    grid_search = False
    basis_scale_factor = 100
    if basis_scale_factor != 1:
        folder_suffix += f'_scale{int(basis_scale_factor)}'
    folder_suffix += log_suffix
    if scale == default_scaler:
        knots = np.linspace(start=-spline_degree / (dof - spline_degree),
                            stop=1 + spline_degree / (dof - spline_degree),
                            num=dof + spline_degree +
                            1)  # take care of the boundary
    else:
        knots = 'equivdist'  # 'equivdist', 'quantile', None
    export_dir = os.path.join(
        log_dir,
        f'{env_class}_value_coverage{folder_suffix}/T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}'
    )

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

    mnar_scheme_list = ['mnar.v0', 'mnar.v0-mar', 'mnar.v1', 'mnar.v1-mar']
    scheme_map = {
        '0': 'no dropout',
        'mnar.v0': 'scheme (a)',
        'mnar.v0-mar': 'scheme (a*)',
        'mnar.v1': 'scheme (b)',
        'mnar.v1-mar': 'scheme (b*)'
    }
    method_map = {'cc': 'CC', 'ipw_propT': 'IPW', 'ipw_propF': 'IPW(est)'}

    np.random.seed(seed=eval_seed)
    if env_class.lower() == 'linear2d' and vectorize_env:
        # specify environment
        low = -norm.ppf(0.999)  # -np.inf
        high = norm.ppf(0.999)  # np.inf
        num_actions = 2
        
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

        # specify eval_S_inits
        eval_S_inits_dict = {}
        for initial_scenario in initial_scenario_list:
            if initial_scenario.lower() == 'a':
                eval_S_inits = np.tile(np.array([[0.5, 0.5]]),
                                       reps=(eval_policy_mc_size, 1))
            elif initial_scenario.lower() == 'b':
                eval_S_inits = np.tile(np.array([[-0.5, -0.5]]),
                                       reps=(eval_policy_mc_size, 1))
            elif initial_scenario.lower() == 'c':
                eval_S_inits = np.random.normal(loc=0,
                                                scale=1,
                                                size=(eval_policy_mc_size,
                                                      env.dim))
            elif initial_scenario.lower() == 'd':
                eval_S_inits = np.clip(np.random.normal(
                    loc=0, scale=1, size=(eval_policy_mc_size, env.dim)),
                                       a_min=-2,
                                       a_max=2)
            elif initial_scenario.lower() == 'e':
                # truncated normal
                eval_S_inits = truncnorm.rvs(-2,
                                             2,
                                             size=(eval_policy_mc_size,
                                                   env.dim))
            eval_S_inits = np.clip(eval_S_inits, a_min=low, a_max=high)
            eval_S_inits_dict[initial_scenario] = eval_S_inits
        print(eval_S_inits_dict)

        # specify policy
        # def target_policy(S):
        #     if S[0] + S[1] > 0:
        #         return [0, 1]
        #     else:
        #         return [1, 0]

        def target_policy(S):
            tmp = np.exp(S[0]+S[1])
            return [1/(1+tmp), tmp/(1+tmp)]

        def vec_target_policy(S):
            if len(S.shape) == 1:
                S = S.reshape(1, -1)
            tmp = np.exp(S[:, 0] + S[:, 1]).flatten()
            return np.column_stack(( 1/(1+tmp), tmp/(1+tmp) ))
            # return np.where(
            #     (S[:, 0] + S[:, 1] > 0).reshape(-1, 1),
            #     np.repeat([[0, 1]], repeats=S.shape[0], axis=0),
            #     np.repeat([[1, 0]], repeats=S.shape[0], axis=0))
        
        policy = vec_target_policy if vectorize_env else target_policy
    else:
        raise NotImplementedError

    if total_N is not None:
        export_dir += '_fixTotal'
    if estimator.lower() != 'cc':
        export_dir += f'_Ubound{round(1/prob_lbound)}'
    if not args.bootstrap:
        filename_true_value = f'{env_class}_true_value_int_gamma{gamma}_size{eval_policy_mc_size}'
        filename_est_value = f'est_value_int_gamma{gamma}'
        filename_CI = f'CI_V_int_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}'  # .pkl
        result_filename = f'result_T_{T}_n_{n}_S_integration_L_{dof}_{estimator}.txt'
        # figure_name = f"est_value_scatterplot_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}.png"
        eval_V_int_CI_multi(
            T=T,
            n=n,
            env=env_dropout,
            L=dof,
            d=spline_degree,
            eval_T=eval_horizon,
            discount=gamma,
            knots=knots,
            total_N=total_N,
            burn_in=burn_in,
            ipw=ipw,
            weight_curr_step=weight_curr_step,
            estimate_missing_prob=estimate_missing_prob,
            prob_lbound=prob_lbound,  # 1e-3, None
            ridge_factor=ridge_factor,
            train_S_inits=None,
            train_S_inits_kwargs={},
            basis_scale_factor=basis_scale_factor,
            vf_mc_size=eval_policy_mc_size,
            mc_size=mc_size,
            alpha_list=alpha_list,
            target_policy=policy,
            dropout_model_type='linear',
            dropout_model = dropout_model,
            dropout_obs_count_thres=dropout_obs_count_thres,
            dropout_scale_obs=False, # True, False
            dropout_include_reward=True, # True, False
            missing_mechanism=missing_mechanism,
            instrument_var_index=instrument_var_index,
            mnar_y_transform=mnar_y_transform,
            psi_init=None if missing_mechanism == 'mnar' and not initialize_with_psiT else psi_true,
            bandwidth_factor=bandwidth_factor,
            value_import_dir=os.path.join(
                log_dir, f'{env_class}_value_coverage{folder_suffix}'),
            export_dir=export_dir,
            filename_true_value=filename_true_value,
            filename_est_value=filename_est_value,
            filename_CI=filename_CI,
            # figure_name=figure_name,
            result_filename=result_filename,
            scale=scale,
            product_tensor=product_tensor,
            eval_env=env,
            eval_S_inits_dict=eval_S_inits_dict,
            eval_seed=eval_seed,
            verbose_freq=1)

        # visualize V_int estimation
        with open(os.path.join(export_dir, filename_est_value), 'rb') as f:
            est_value_dict = pickle.load(f)
        initial_states = est_value_dict['initial_states']
        est_value_int = est_value_dict['est_V_int_list']
        est_value_int_std = est_value_dict['est_V_int_std_list']
        true_value_int = est_value_dict['true_V_int']
        est_V_mse = est_value_dict['est_V_mse_list']
        max_inverse_wt_list = est_value_dict['max_inverse_wt_list']
        mnar_psi_est_list = est_value_dict['mnar_psi_est_list']
        dropout_prob_mse_list = est_value_dict['dropout_prob_mse_list']
        for initial_scenario in initial_scenario_list:
            nrows, ncols = 1, 2
            fig, ax = plt.subplots(nrows=nrows,
                                   ncols=ncols,
                                   figsize=(4 * ncols, 4 * nrows))
            sns.histplot(data=est_value_int[initial_scenario],
                         ax=ax[0],
                         stat='probability')
            ax[0].axvline(true_value_int[initial_scenario], color='red')
            ax[0].set_title(
                f'V hat ({scheme_map[dropout_scheme]}, {estimator}, n={n})')
            sns.histplot(data=est_value_int_std[initial_scenario],
                         ax=ax[1],
                         stat='probability')
            ax[1].set_title(
                f'V hat std ({scheme_map[dropout_scheme]}, {estimator}, n={n})'
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    export_dir,
                    f"value_est_dist_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}_init_{initial_scenario}.png"
                ))
            plt.close()
            # scatterplot: max inverse weight vs. MSE
            if ipw:
                ax = sns.scatterplot(x=est_V_mse[initial_scenario],
                                     y=max_inverse_wt_list)
                ax.set_title(
                    f'scheme {scheme_map[dropout_scheme]}, {method_map[estimator]}, n={n}'
                )
                ax.set_xlabel('V mse')
                ax.set_ylabel('maximum inverse weight')
                plt.savefig(
                    os.path.join(
                        export_dir,
                        f"value_vs_maxInvWt_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}_init_{initial_scenario}.png"
                    ))
                plt.close()
    else:
        filename_true_value = f'{env_class}_true_value_int_gamma{gamma}_size{eval_policy_mc_size}'
        filename_est_value = f'est_value_int_gamma{gamma}'
        filename_CI = f'CI_V_int_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}'  # .pkl
        filename_bootstrap_CI = f'bootstrap_CI_V_int_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}'  # .pkl
        result_filename = f'result_T_{T}_n_{n}_S_integration_L_{dof}_{estimator}.txt'
        # figure_name = f"est_value_scatterplot_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}.png"
        eval_V_int_CI_bootstrap_multi(
            T=T,
            n=n,
            env=env_dropout,
            L=dof,
            d=spline_degree,
            eval_T=eval_horizon,
            discount=gamma,
            knots=knots,
            total_N=total_N,
            burn_in=burn_in,
            ipw=ipw,
            weight_curr_step=weight_curr_step,
            estimate_missing_prob=estimate_missing_prob,
            prob_lbound=prob_lbound,  # 1e-3, None
            ridge_factor=ridge_factor,
            train_S_inits=None,
            train_S_inits_kwargs={},
            basis_scale_factor=basis_scale_factor,
            vf_mc_size=eval_policy_mc_size,
            mc_size=mc_size,
            alpha_list=alpha_list,
            bootstrap_size=25,
            target_policy=policy,
            dropout_model_type='linear',
            dropout_model = dropout_model,
            dropout_obs_count_thres=dropout_obs_count_thres,
            dropout_scale_obs=False, # True, False
            dropout_include_reward=True, # True, False
            missing_mechanism=missing_mechanism,
            instrument_var_index=instrument_var_index,
            mnar_y_transform=mnar_y_transform,
            psi_init=None if missing_mechanism == 'mnar'
            and not initialize_with_psiT else psi_true,
            bandwidth_factor=bandwidth_factor,
            value_import_dir=os.path.join(
                log_dir, f'{env_class}_value_coverage{folder_suffix}'),
            export_dir=export_dir,
            filename_true_value=filename_true_value,
            filename_est_value=filename_est_value,
            filename_CI=filename_CI,
            filename_bootstrap_CI=filename_bootstrap_CI,
            # figure_name=figure_name,
            result_filename=result_filename,
            scale=scale,
            product_tensor=product_tensor,
            eval_env=env,
            eval_S_inits_dict=eval_S_inits_dict,
            eval_seed=eval_seed,
            verbose_freq=1)

        with open(os.path.join(export_dir, filename_est_value), 'rb') as f:
            est_value_dict = pickle.load(f)
        initial_states = est_value_dict['initial_states']
        est_value_int = est_value_dict['est_V_int_list']
        est_value_int_std = est_value_dict['est_V_int_std_list']
        bootstrap_value_int_std = est_value_dict['bootstrap_V_int_std_list']
        true_value_int = est_value_dict['true_V_int']
        est_V_mse = est_value_dict['est_V_mse_list']
        max_inverse_wt_list = est_value_dict['max_inverse_wt_list']
        mnar_psi_est_list = est_value_dict['mnar_psi_est_list']
        dropout_prob_mse_list = est_value_dict['dropout_prob_mse_list']
        # histograms to visualize V_int estimation
        for initial_scenario in initial_scenario_list:
            nrows, ncols = 1, 2
            fig, ax = plt.subplots(nrows=nrows,
                                   ncols=ncols,
                                   figsize=(4 * ncols, 4 * nrows))
            sns.histplot(data=est_value_int[initial_scenario],
                         ax=ax[0],
                         stat='probability')
            ax[0].axvline(true_value_int[initial_scenario], color='red')
            ax[0].set_title(
                f'V hat ({scheme_map[dropout_scheme]}, {estimator}, n={n})')
            sns.histplot(data=bootstrap_value_int_std[initial_scenario],
                         ax=ax[1],
                         stat='probability')
            ax[1].axvline(np.std(est_value_int[initial_scenario], ddof=1),
                          color='red')
            ax[1].set_title(
                f'V hat std ({scheme_map[dropout_scheme]}, {estimator}, n={n})'
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    export_dir,
                    f"value_est_dist_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}_init_{initial_scenario}.png"
                ))
            plt.close()
            # scatterplot: max inverse weight vs. MSE
            if ipw:
                ax = sns.scatterplot(x=est_V_mse[initial_scenario],
                                     y=max_inverse_wt_list)
                ax.set_title(
                    f'scheme {scheme_map[dropout_scheme]}, {method_map[estimator]}, n={n}'
                )
                ax.set_xlabel('V mse')
                ax.set_ylabel('maximum inverse weight')
                plt.savefig(
                    os.path.join(
                        export_dir,
                        f"value_vs_maxInvWt_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}_init_{initial_scenario}.png"
                    ))
                plt.close()

    if ipw and estimate_missing_prob and missing_mechanism == 'mnar':
        mnar_psi_est_list = np.hstack(mnar_psi_est_list)
        nrows, ncols = 1, 2
        fig, ax = plt.subplots(nrows=nrows,
                               ncols=ncols,
                               figsize=(4 * ncols, 4 * nrows))
        sns.histplot(data=mnar_psi_est_list, stat='probability', ax=ax[0])
        ax[0].axvline(1.5, color='red')
        ax[0].set_title(f'psi est')
        sns.histplot(data=dropout_prob_mse_list, stat='probability', ax=ax[1])
        ax[1].set_title(f'prob est mse')
        plt.savefig(
            os.path.join(
                export_dir,
                f"mnar_est_psi_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}.png"
            ))
        plt.close()