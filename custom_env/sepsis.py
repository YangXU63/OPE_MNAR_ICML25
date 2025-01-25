import numpy as np
import gym
from gym.spaces import Box, Discrete
from gym.vector import VectorEnv
from gym.utils import seeding

from ope_mnar.utils import constant_fn

class SepsisVectorEnvSpec(VectorEnv):

    def __init__(self,
                 num_envs,
                 T=20,
                 static_state_list=[],
                 dynamic_state_list=[],
                 action_levels=None,
                 vec_state_trans_model=None,
                 vec_reward_model=None,
                 vec_dropout_model=None,
                 low=-np.inf,
                 high=np.inf,
                 dtype=np.float32):

        action_space = Discrete(n=action_levels)
        state_list = static_state_list + dynamic_state_list
        state_dim = len(state_list)
        observation_space = Box(low=low,
                                high=high,
                                shape=(state_dim, ),
                                dtype=dtype)
        super().__init__(num_envs, observation_space, action_space)

        self.is_vector_env = True
        self.observations = None
        self.dim = state_dim
        self.T = T  # max length of trajectory
        self._np_random = np.random
        self.last_obs = None
        self.low = low
        self.high = high
        if vec_state_trans_model is not None:
            assert callable(vec_state_trans_model)
            self.state_trans_model = vec_state_trans_model
        else:
            self.state_trans_model = None
        if vec_reward_model:
            assert callable(vec_reward_model)
            self.reward_model = vec_reward_model
        else:
            self.reward_model = None
        if vec_dropout_model is not None:
            assert callable(vec_dropout_model)
        else:
            vec_dropout_model = constant_fn(val=0)
        self.dropout_model = vec_dropout_model
        self.seed()

    def reset_wait(self, timeout=None):
        """
		Parameters
		----------
		timeout : int or float, optional
			Number of seconds before the call to `reset_wait` times out. If
			`None`, the call to `reset_wait` never times out.
		Returns
		-------
		observations : sample from `observation_space`
			A batch of observations from the vectorized environment.
		"""
        return self.observations

    def reset(self, S_inits=None):
        r"""Reset all sub-environments and return a batch of initial observations.
        
        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.
        """
        self.reset_async(S_inits)
        return self.reset_wait()

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    @property
    def np_random(self):
        """Lazily seed the rng since this is expensive and only needed if
        sampling from this space.
        """
        if self._np_random is None:
            self.seed()
        return self._np_random


