import numpy as np
from functools import partial
from ope_mnar.utils import SimEnv, VectorSimEnv


def state_trans_model(obs, action, rng=np.random):
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    trans_mat = np.array([[(2 * action - 1), 0], [0, (1 - 2 * action)]])
    bias = np.array([0, 0])
    noise = rng.normal(loc=0, scale=0.5, size=trans_mat.shape[0])
    S_next = np.dot(obs, trans_mat) + bias + noise
    return S_next


def vec_state_trans_model(obs, action, rng=np.random):
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    if len(obs.shape) == 1:
        obs = obs.reshape(1, -1)
    if not isinstance(action, np.ndarray):
        action = np.array(action)
    if not action.shape or len(action.shape) == 1:
        action = action.reshape(-1, 1)

    obs_action = np.concatenate([obs, action, obs * action], axis=1)
    trans_mat = np.array([[-1, 0], [0, 1], [0, 0], [2, 0], [0, -2]])
    S_next = np.matmul(obs_action, trans_mat).astype('float')
    S_next += rng.normal(loc=0.0, scale=0.5, size=S_next.shape)
    return S_next.squeeze()


def reward_model(obs, action, next_obs, rng=np.random):
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    if not isinstance(next_obs, np.ndarray):
        next_obs = np.array(next_obs)
    next_weight = np.array([2, 1])
    weight = np.array([0, 0.5])
    bias = -(2 * action - 1) / 4
    noise = rng.normal(loc=0.0, scale=0.01)
    reward = np.dot(next_obs, next_weight) + np.dot(obs, weight) + bias + noise
    # reward = np.dot(obs, weight) + bias + noise
    return reward


def vec_reward_model(obs, action, next_obs, rng=np.random):
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    if len(obs.shape) == 1:
        obs = obs.reshape(1, -1)
    if not isinstance(action, np.ndarray):
        action = np.array(action)
    if not action.shape or len(action.shape) == 1:
        action = action.reshape(-1, 1)
    if not isinstance(next_obs, np.ndarray):
        next_obs = np.array(next_obs)
    if len(next_obs.shape) == 1:
        next_obs = next_obs.reshape(1, -1)
    obs_nobs_action = np.concatenate([obs, next_obs, action], axis=1)
    weight = np.array([0, 0.5, 2, 1, -1 / 2]).reshape(-1, 1)
    # weight = np.array([0, 0.5, 0,0,-1 / 2]).reshape(-1, 1)
    reward = np.matmul(obs_nobs_action, weight)
    bias = np.zeros_like(reward) + 1 / 4
    reward += bias
    noise = rng.normal(loc=0.0, scale=0.01, size=reward.shape)
    reward += noise
    return reward


def dropout_model(obs_history, action_history, reward_history, T,
                  dropout_scheme, dropout_rate, dropout_obs_count_thres):
    if len(action_history) < dropout_obs_count_thres:
        return 0
    intercept = 7  # 3
    if dropout_scheme == '0':
        return 0
    elif dropout_scheme == 'mnar.v0':
        if T == 25:
            if dropout_rate == 0.9:
                if dropout_obs_count_thres == 5:
                    logit = intercept + 2.4 + 0.8 * obs_history[-2][
                        0] - 1.5 * reward_history[-1]
                elif dropout_obs_count_thres == 2:
                    logit = intercept + 3.2 + 0.8 * obs_history[-2][
                        0] - 1.5 * reward_history[-1]
                elif dropout_obs_count_thres == 1:
                    logit = intercept + 4 + 0.8 * obs_history[-2][
                        0] - 1.5 * reward_history[-1]
            else:
                raise NotImplementedError
        prob = 1 / (np.exp(logit) + 1)
    else:
        raise NotImplementedError
    return prob


def vec_dropout_model(obs_history, action_history, reward_history, T,
                      dropout_scheme, dropout_rate, dropout_obs_count_thres):
    if not isinstance(obs_history, np.ndarray):
        obs_history = np.array(obs_history)
    if not isinstance(action_history, np.ndarray):
        action_history = np.array(action_history)
    if not isinstance(reward_history, np.ndarray):
        reward_history = np.array(reward_history)
    # dropout only happens after a threshold
    # if action_history.shape[1] <= dropout_obs_count_thres: # burn_in
    if action_history.shape[1] < dropout_obs_count_thres:
        return np.zeros(shape=obs_history.shape[0]).reshape(-1, 1)
    intercept = -1  # 3
    if dropout_scheme == '0':
        prob = np.zeros(shape=obs_history.shape[0])
    elif dropout_scheme == 'mar.v0':
        if T == 25:
            if dropout_rate == 0.6:
                logit = intercept + 4.5 + 0.8 * obs_history[:, -2,
                                                            0] - 1.5 * reward_history[:,
                                                                                      -2]
            elif dropout_rate == 0.7:
                logit = intercept + 3.2 + 0.8 * obs_history[:, -2,
                                                            0] - 1.5 * reward_history[:,
                                                                                      -2]
            elif dropout_rate == 0.8:
                logit = intercept + 2.5 + 0.8 * obs_history[:, -2,
                                                            0] - 1.5 * reward_history[:,
                                                                                      -2]
            elif dropout_rate == 0.9:
                if dropout_obs_count_thres == 5:
                    logit = intercept + 1.5 + 0.8 * obs_history[:, -2,
                                                                0] - 1.5 * reward_history[:,
                                                                                          -2]
                elif dropout_obs_count_thres == 2:
                    # logit = intercept + 2.3 + 0.8 * obs_history[:, -2,
                    #                                             0] - 1.5 * reward_history[:,
                    #                                                                       -2]
                    logit = intercept + 3.2 + 0.15 * obs_history[:, -2,
                                                                0] - 0.3 * reward_history[:,
                                                                                          -2] 
                elif dropout_obs_count_thres == 1:
                    logit = intercept + 2.8 + 0.8 * obs_history[:, -2,
                                                                0] - 1.5 * reward_history[:,
                                                                                          -2]
            else:
                raise NotImplementedError
        elif T == 10:
            if dropout_rate == 0.6:
                if dropout_obs_count_thres == 5:
                    logit = intercept + 0. + 0.8 * obs_history[:, -2,
                                                               0] - 1.5 * reward_history[:,
                                                                                         -2]
                elif dropout_obs_count_thres == 2:
                    logit = intercept + 3.2 + 0.15 * obs_history[:, -2,
                                                                0] - 0.3 * reward_history[:,
                                                                                          -2]
            elif dropout_rate == 0.7:
                logit = intercept - 1. + 0.8 * obs_history[:, -2,
                                                           0] - 1.5 * reward_history[:,
                                                                                     -2]
            elif dropout_rate == 0.8:
                logit = intercept - 2. + 0.8 * obs_history[:, -2,
                                                           0] - 1.5 * reward_history[:,
                                                                                     -2]
            elif dropout_rate == 0.9:
                logit = intercept + 3.2 + 0.15 * obs_history[:, -2,
                                                           0] - 0.3 * reward_history[:,
                                                                                     -2]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        prob = 1 / (np.exp(logit) + 1)
    elif dropout_scheme == 'mar.v3':
        logit = intercept + 3.5 + 0.8 * obs_history[:, -2, 0] - 1.5 * reward_history[:,-2]
        prob = 1 / (np.exp(logit) + 1)
    elif dropout_scheme == 'mnar.v0':
        if T == 25:
            if dropout_rate == 0.6:
                logit = intercept + 7.0 + 0.8 * obs_history[:, -2,
                                                            0] - 1.5 * reward_history[:,
                                                                                      -1]
            elif dropout_rate == 0.7:
                logit = intercept + 5.0 + 0.8 * obs_history[:, -2,
                                                            0] - 1.5 * reward_history[:,
                                                                                      -1]
            elif dropout_rate == 0.8:
                logit = intercept + 3.6 + 0.8 * obs_history[:, -2,
                                                            0] - 1.5 * reward_history[:,
                                                                                      -1]
            elif dropout_rate == 0.9:
                if dropout_obs_count_thres == 5:
                    logit = intercept + 2.4 + 0.8 * obs_history[:, -2,
                                                                0] - 1.5 * reward_history[:,
                                                                                          -1]
                elif dropout_obs_count_thres == 2:
                    logit = intercept + 3.2 + 0.15 * obs_history[:, -2,
                                                                0] - 0.3 * reward_history[:,
                                                                                          -1]
                    # logit = intercept + 3.2 + 0.1 * obs_history[:, -2,
                    #                                             0] - 0.5 * reward_history[:,
                    #                                                                       -1]
                elif dropout_obs_count_thres == 1:
                    logit = intercept + 4 + 0.8 * obs_history[:, -2,
                                                              0] - 1.5 * reward_history[:,
                                                                                        -1]
            else:
                raise NotImplementedError
        elif T == 10:
            if dropout_rate == 0.6:
                if dropout_obs_count_thres == 5:
                    logit = intercept + 2 + 0.8 * obs_history[:, -2,
                                                              0] - 1.5 * reward_history[:,
                                                                                        -1]
                elif dropout_obs_count_thres == 3:
                    logit = intercept + 3 + 0.8 * obs_history[:, -2,
                                                              0] - 1.5 * reward_history[:,
                                                                                        -1]
                elif dropout_obs_count_thres == 2:
                    logit = intercept + 3.2 + 0.15 * obs_history[:, -2,
                                                              0] - 0.3 * reward_history[:,
                                                                                        -1]
            elif dropout_rate == 0.7:
                logit = intercept + 0. + 0.8 * obs_history[:, -2,
                                                           0] - 1.5 * reward_history[:,
                                                                                     -1]
            elif dropout_rate == 0.8:
                logit = intercept - 0.8 + 0.8 * obs_history[:, -2,
                                                            0] - 1.5 * reward_history[:,
                                                                                      -1]
            elif dropout_rate == 0.9:
                if dropout_obs_count_thres == 6:
                    logit = intercept - 2. + 0.8 * obs_history[:, -2,
                                                               0] - 1.5 * reward_history[:,
                                                                                         -1]
                elif dropout_obs_count_thres == 5:
                    logit = intercept - 1.2 + 0.8 * obs_history[:, -2,
                                                                0] - 1.5 * reward_history[:,
                                                                                          -1]
                elif dropout_obs_count_thres == 4:
                    logit = intercept - 0.9 + 0.8 * obs_history[:, -2,
                                                                0] - 1.5 * reward_history[:,
                                                                                          -1]
                elif dropout_obs_count_thres == 3:
                    logit = intercept - 0.6 + 0.8 * obs_history[:, -2,
                                                                0] - 1.5 * reward_history[:,
                                                                                          -1]
                elif dropout_obs_count_thres == 2:
                    logit = intercept + 3.2 + 0.15 * obs_history[:, -2,
                                                                0] - 0.3 * reward_history[:,
                                                                                          -1]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        prob = 1 / (np.exp(logit) + 1)
    elif dropout_scheme == 'mnar.v1':
        obs = obs_history[:, -2, :]
        reward = reward_history[:, -1]
        logit = intercept + 4 - 0.5 * np.power(obs[:, 0], 2) - 1.5 * reward
        prob = 1 / (np.exp(logit) + 1)
    elif dropout_scheme == 'mnar.v2':
        obs = obs_history[:, -2, :]
        reward = reward_history[:, -1]
        # logit = intercept + 5 + 0.5 * obs[:, 0] - 1. * reward - 0.1 * (reward ** 2)
        logit = intercept + 4 + 0.5 * obs[:, 0] - 1. * reward - 0.1 * (reward ** 2)
        prob = 1 / (np.exp(logit) + 1)
    elif dropout_scheme == 'mnar.v3':
        # obs = obs_history[:, -2, :]
        # reward = reward_history[:, -1]
        # logit = intercept + 5 - 0.5 * np.power(obs[:, 0], 2) - 1. * reward - 0.2 * (reward ** 2)
        logit = intercept + 3.5 + 0.8 * obs_history[:, -2, 0] - 1.5 * reward_history[:,-1]
        prob = 1 / (np.exp(logit) + 1)
    else:
        raise NotImplementedError
    return prob.reshape(-1, 1)


def dropout_model_0(obs_history, action_history, reward_history):
    return 0

def vec_dropout_model_0(obs_history, action_history, reward_history):
    return np.zeros(shape=(obs_history.shape[0], 1))

class Linear2dEnv(SimEnv):

    def __init__(
        self,
        T,
        dropout_scheme='0',
        dropout_rate=0.,
        dropout_obs_count_thres=1,
        low=-np.inf,
        high=np.inf,
        seed=0
    ):
        if dropout_scheme == '0' or dropout_rate == 0:
            partial_vec_dropout_model = dropout_model_0
        else:
            partial_vec_dropout_model = partial(
                dropout_model,
                T=T,
                dropout_scheme=dropout_scheme,
                dropout_rate=dropout_rate,
                dropout_obs_count_thres=dropout_obs_count_thres)

        super().__init__(T=T,
                         dim=2,
                         num_actions=2,
                         state_trans_model=state_trans_model,
                         reward_model=reward_model,
                         dropout_model=partial_vec_dropout_model,
                         low=low,
                         high=high,
                         dtype=np.float32,
                         seed=seed)


class Linear2dVectorEnv(VectorSimEnv):

    def __init__(
        self,
        num_envs,
        T,
        dropout_rate=0.,
        dropout_scheme='0',
        dropout_obs_count_thres=1,
        low=-np.inf,
        high=np.inf,
        seed=0
    ):
        if dropout_scheme == '0' or dropout_rate == 0:
            partial_vec_dropout_model = vec_dropout_model_0
        else:
            partial_vec_dropout_model = partial(
                vec_dropout_model,
                T=T,
                dropout_scheme=dropout_scheme,
                dropout_rate=dropout_rate,
                dropout_obs_count_thres=dropout_obs_count_thres)

        super().__init__(num_envs=num_envs,
                         T=T,
                         dim=2,
                         num_actions=2,
                         vec_state_trans_model=vec_state_trans_model,
                         vec_reward_model=vec_reward_model,
                         vec_dropout_model=partial_vec_dropout_model,
                         low=low,
                         high=high,
                         dtype=np.float32,
                         seed=seed)

        # self._dropout_rate = dropout_rate
        # self._dropout_scheme = dropout_scheme
        # self._dropout_obs_count_thres = dropout_obs_count_thres