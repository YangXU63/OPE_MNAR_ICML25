import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle
import pathlib

try:
    from batch_rl.dqn import DQN, DuelingDQN
    from batch_rl.bcq import discrete_BCQ
    from batch_rl.rem import REM
    from batch_rl.utils import ReplayBuffer, ReplayBufferPER
    from ope_mnar.utils import MinMaxScaler
except:
    import sys
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar/batch_rl'))
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar'))
    from batch_rl.dqn import DQN, DuelingDQN
    from batch_rl.bcq import discrete_BCQ
    from batch_rl.rem import REM
    from batch_rl.utils import ReplayBuffer, ReplayBufferPER
    from ope_mnar.utils import MinMaxScaler


parser = argparse.ArgumentParser()
parser.add_argument('--discount', type=float, default=0.8)
parser.add_argument('--RL_agent', type=str, default='dqn') # 'dqn', 'bcq', 'rem', 'dueling-dqn'
parser.add_argument('--prioritized_replay', type=lambda x: (str(x).lower() == 'true'), default=True)   
parser.add_argument('--max_iters', type=int, default=int(2e5)) # int(2e5)
parser.add_argument('--minibatch_size', type=int, default=256) # 256
parser.add_argument('--lr', type=float, default=1e-3) # 1e-3
args = parser.parse_args()

if __name__ == '__main__':
    discount = args.discount
    RL_agent = args.RL_agent
    prioritized_replay = args.prioritized_replay
    max_iters = args.max_iters
    minibatch_size = args.minibatch_size
    lr = args.lr
    subsample_size = None # None, 500
    data_dir = os.path.expanduser('~/Data/mimic-iii')
    export_dir = os.path.expanduser('~/Projects/ope_mnar/output/sepsis/batch_rl')
    pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
    
    shift_bloc = False # True
    exclude_icu_morta = False # True
    use_complete_trajs = False
    # shift_bloc_str = '_shiftbloc' if shift_bloc else ''
    # exclude_icu_morta_str = '_exclude_icu_morta' if exclude_icu_morta else ''
    full_trajs_str = '_full_trajs' if use_complete_trajs else ''
    filename_suffix = '' # '', '_realigned_lvcf', '_realigned'
    filename_suffix += '_exclude_icu_morta' if exclude_icu_morta else ''
    filename_suffix += '_shiftbloc' if shift_bloc else ''

    ########################################################################
    ##                    Specify state features
    ########################################################################
    ## all features
    with open(os.path.join(data_dir, 'state_features.txt')) as f:
        features_cols = f.read().split()

    ## selected features
    # features_cols.remove('re_admission') # 're_admission' are all 0's
    # features_cols = ['Arterial_pH','SpO2','Temp_C','Chloride','Hb','INR','age','PT','HR','Arterial_BE','Ionised_Ca','Calcium','Arterial_lactate'] + ['SOFA'] # 14 features
    features_cols = ['Arterial_pH','SpO2','Temp_C','Chloride','Hb','INR','age','PT','HR','Arterial_BE','Ionised_Ca','Calcium','Arterial_lactate','SOFA','RR'] # 15 features
    static_features = [f for f in features_cols if f in set(['gender', 'age', 'Weight_kg', 're_admission'])]
    dynamic_features = [f for f in features_cols if f not in static_features]
    features_cols = static_features + dynamic_features # should keep this order in all downstream analysis
    state_dim = len(features_cols)

    ########################################################################
    ##                    Specify actions
    ########################################################################
    action_col = 'action' # already discretized
    num_actions = 3

    ########################################################################
    ##                    Process data then add to bufffer
    ########################################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if subsample_size is None:
        # data_path = os.path.join(data_dir, f'sepsis_processed{full_trajs_str}_state_action{num_actions}_reward{exclude_icu_morta_str}{shift_bloc_str}_split1.csv')
        data_path = os.path.join(data_dir, f'sepsis_processed{full_trajs_str}_state_action{num_actions}_reward{filename_suffix}_split1.csv')
    else:
        # data_path = os.path.join(data_dir, f'sepsis_processed{full_trajs_str}_state_action{num_actions}_reward{exclude_icu_morta_str}{shift_bloc_str}_split1_sample{subsample_size}.csv')
        data_path = os.path.join(data_dir, f'sepsis_processed{full_trajs_str}_state_action{num_actions}_reward{filename_suffix}_split1_sample{subsample_size}.csv')
    data = pd.read_csv(data_path)

    ## scale features
    scaler_path = os.path.join(data_dir, f'feature{state_dim}_scaler{filename_suffix}.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path,'rb') as f:
            scaler = pickle.load(f)
    else:
        # full_data = pd.read_csv(os.path.join(data_dir, f'sepsis_processed_state_action{num_actions}_reward{exclude_icu_morta_str}{shift_bloc_str}.csv'))
        full_data = pd.read_csv(os.path.join(data_dir, f'sepsis_processed_state_action{num_actions}_reward{filename_suffix}.csv'))
        scaler = MinMaxScaler()
        scaler.fit(full_data[features_cols])
        with open(scaler_path,'wb') as f:
            pickle.dump(scaler, f)

    print(pd.DataFrame({'Variable': features_cols, 'Min': scaler.data_min_, 'Max': scaler.data_max_}))
    data[features_cols] = scaler.transform(data[features_cols])
    print(data[features_cols].head())
    assert not data[features_cols].isna().any().any(), 'There is missing data in the table.'
    
    ########################################################################
    ##                    Specify rewards
    ########################################################################
    # custom_reward_name = 'raghu'
    
    # 2) custom reward function
    custom_reward_name = 'raghu_v1'
    c0 = -0.5
    c1 = -0.25
    c2 = -1
    c3 = 1

    def custom_reward(rows):
        next_sofa = rows['SOFA'].iloc[1:].values
        curr_sofa = rows['SOFA'].iloc[:-1].values
        next_lactate = rows['Arterial_lactate'].iloc[1:].values
        curr_lactate = rows['Arterial_lactate'].iloc[:-1].values
        reward = ((next_sofa == curr_sofa) & (next_sofa > 0)) * c0 + (next_sofa - curr_sofa) * c1 \
            + np.tanh(next_lactate - curr_lactate) * c2 + c3
        reward_full = np.append(reward, 0)
        rows['reward_' + custom_reward_name] = reward_full
        return rows

    # # 3) use negative SOFA score as reward
    # custom_reward_name = 'neg_sofa'

    # def custom_reward(rows):
    #     next_sofa = rows['SOFA'].iloc[1:].values
    #     reward = 0.1 * (23 - next_sofa)
    #     reward_full = np.append(reward, 0)
    #     rows['reward_'+custom_reward_name] = reward_full
    #     return rows

    data = data.groupby('icustayid').apply(custom_reward)

    if custom_reward_name == 'raghu':   
        reward_col = 'reward_raghu' # without the dominant term at terminal step
    else:
        reward_col = 'reward_' + custom_reward_name

    ########################################################################
    ##                    Load data to replay buffer
    ########################################################################
    print('Load data into buffer...')
    id_col = 'icustayid'
    id_list = data[id_col].unique()
    T = data.groupby(id_col).size().max()
    print(f'Number of trajectories: {len(id_list)}')
    if prioritized_replay:
        buffer = ReplayBufferPER(state_dim=state_dim, buffer_size=data.shape[0], device=device)
    else:
        buffer = ReplayBuffer(state_dim=state_dim, buffer_size=data.shape[0], device=device)
    initial_states = []
    for i, id_ in enumerate(id_list):
        tmp = data.loc[data[id_col] == id_]
        traj_len = len(tmp)
        initial_states.append(tmp[features_cols].iloc[0].values)
        if traj_len < T:
            states = tmp[features_cols].iloc[:-1].values
            next_states = tmp[features_cols].iloc[1:].values
            actions = tmp[action_col].iloc[:-1].values.reshape(-1,1)
            rewards = tmp[reward_col].iloc[:-1].values.reshape(-1,1)
            dones = np.zeros(shape=rewards.shape)
        else:
            states = tmp[features_cols].values
            next_states = np.vstack([tmp[features_cols].iloc[1:].values, np.zeros(shape=(1, state_dim))]) # use 0's to pad the last observation
            actions = tmp[action_col].values.reshape(-1,1)
            rewards = tmp[reward_col].values.reshape(-1,1)
            dones = np.zeros(shape=rewards.shape)
            dones[-1] = 1
        if prioritized_replay:
            buffer.add_episode(states, actions, next_states, rewards, dones, priorities=None)
        else:
            buffer.add_episode(states, actions, next_states, rewards, dones)

    initial_states = np.vstack(initial_states)
    print('Finished!')
        
    ########################################################################
    ##                    Train Agent
    ########################################################################
    print(f'agent: {RL_agent.lower()}')
    if RL_agent.lower() == 'dqn':
        ## train optimal policy
        dqn = DQN(num_actions=num_actions,
            state_dim=state_dim,
            device=device,
            discount=discount,
            optimizer="Adam",
            optimizer_parameters={'lr': lr},
            polyak_target_update=True,
            target_update_frequency=1e3,
            tau=0.005)
        dqn.train(replay_buffer=buffer, max_iters=max_iters, minibatch_size=minibatch_size, verbose=True)
        # evaluate learned policy
        V_int = dqn.evaluate(initial_states=initial_states)
        print(f'estimated value: {V_int}')
        # track Q loss
        fig = dqn.plot_loss(smooth_window=1000)
        if prioritized_replay:
            fig.savefig(os.path.join(export_dir, f'{RL_agent}_per_state{state_dim}_act{num_actions}_reward_{custom_reward_name}_loss.png'))
            dqn.save(filename=os.path.join(export_dir, f'{RL_agent}_per_state{state_dim}_act{num_actions}_reward_{custom_reward_name}'))
        else:
            fig.savefig(os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}_loss.png'))
            dqn.save(filename=os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}'))
    elif RL_agent.lower() == 'dueling-dqn':
        # train optimal policy
        dqn = DuelingDQN(num_actions=num_actions,
            state_dim=state_dim,
            device=device,
            discount=discount,
            optimizer="Adam",
            optimizer_parameters={'lr': lr},
            polyak_target_update=True,
            target_update_frequency=1e3,
            tau=0.005)
        dqn.train(replay_buffer=buffer, max_iters=max_iters, minibatch_size=minibatch_size, verbose=True)
        # evaluate learned policy
        V_int = dqn.evaluate(initial_states=initial_states)
        print(f'estimated value: {V_int}')
        # track Q loss
        fig = dqn.plot_loss(smooth_window=1000)
        if prioritized_replay:
            fig.savefig(os.path.join(export_dir, f'{RL_agent}_per_state{state_dim}_act{num_actions}_reward_{custom_reward_name}_loss.png'))
            dqn.save(filename=os.path.join(export_dir, f'{RL_agent}_per_state{state_dim}_act{num_actions}_reward_{custom_reward_name}'))
        else:
            fig.savefig(os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}_loss.png'))
            dqn.save(filename=os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}'))
    elif RL_agent.lower() == 'bcq':
        # train optimal policy
        bcq = discrete_BCQ(num_actions=num_actions,
            state_dim=state_dim,
            device=device,
            BCQ_threshold=0.3,
            discount=discount,
            optimizer="Adam",
            optimizer_parameters={'lr': lr},
            polyak_target_update=True,
            target_update_frequency=1e3,
            tau=0.005)
        bcq.train(replay_buffer=buffer, max_iters=max_iters, minibatch_size=minibatch_size, verbose=True)
        # evaluate learned policy
        V_int = bcq.evaluate(initial_states=initial_states)
        print(f'estimated value: {V_int}')
        # track Q loss
        fig = bcq.plot_loss(smooth_window=1000)
        fig.savefig(os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}_loss.png'))
        bcq.save(filename=os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}'))
    elif RL_agent.lower() == 'rem':
        # train optimal policy
        rem = REM(num_actions=num_actions,
            state_dim=state_dim,
            num_networks=4, 
            device=device,
            transform_strategy='stochastic',
            num_convex_combinations=1,
            discount=discount,
            optimizer="Adam",
            optimizer_parameters={'lr': lr},
            polyak_target_update=True,
            target_update_frequency=1e3,
            tau=0.005)
        rem.train(replay_buffer=buffer, max_iters=max_iters, minibatch_size=minibatch_size, verbose=True)
        # evaluate learned policy
        V_int = rem.evaluate(initial_states=initial_states)
        print(f'estimated value: {V_int}')
        # track Q loss
        fig = rem.plot_loss(smooth_window=1000)
        fig.savefig(os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}_loss.png'))
        rem.save(filename=os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}'))
    else:
        raise NotImplementedError

