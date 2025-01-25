
"""
Classic cart-pole system implemented by Rich Sutton et al.
Code modified from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""
import math
from typing import Optional, Union, Callable
from functools import partial
import numpy as np

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from gym.utils import seeding
from gym.vector import VectorEnv

try:
    from ope_mnar.utils import constant_fn, constant_vec_fn
except:
    import os
    import sys
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar/ope_mnar'))
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar'))
    from ope_mnar.utils import constant_fn, constant_vec_fn

def dropout_model(obs_history, action_history, reward_history, T,
                  dropout_scheme=None, dropout_rate=None, dropout_obs_count_thres=None):
    if len(action_history) < dropout_obs_count_thres:
        return 0
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4
    if dropout_scheme == '0':
        prob = 0
    elif dropout_scheme == 'mnar.v0':
        x, x_dot, theta, theta_dot = obs_history[-1, :].T
        terminated = np.logical_or.reduce((x < -x_threshold, x > x_threshold, 
                theta < -theta_threshold_radians, theta > theta_threshold_radians)).astype(bool)
        if dropout_rate == 0.9:
            logit = 7 - 3 * terminated
        elif dropout_rate == 0.6:
            logit = 7 - 2.2 * terminated
        prob = 1 / (np.exp(logit) + 1)
    elif dropout_scheme == 'mar.v0':
        x, x_dot, theta, theta_dot = obs_history[-2, :].T
        terminated = np.logical_or.reduce((x < -x_threshold, x > x_threshold, 
                theta < -theta_threshold_radians, theta > theta_threshold_radians)).astype(bool)
        if dropout_rate == 0.9:
            logit = 7 - 3 * terminated
        elif dropout_rate == 0.6:
            logit = 7 - 2.2 * terminated
        prob = 1 / (np.exp(logit) + 1)
    else:
        raise NotImplementedError
    return prob

def vec_dropout_model(obs_history, action_history, reward_history, T,
                      dropout_scheme=None, dropout_rate=None, dropout_obs_count_thres=None):
    if not isinstance(obs_history, np.ndarray):
        obs_history = np.array(obs_history)
    if not isinstance(action_history, np.ndarray):
        action_history = np.array(action_history)
    if not isinstance(reward_history, np.ndarray):
        reward_history = np.array(reward_history)
    # dropout only happens after a threshold
    if action_history.shape[1] < dropout_obs_count_thres:
        return np.zeros(shape=obs_history.shape[0]).reshape(-1, 1)
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4
    if dropout_scheme == '0':
        prob = np.zeros(shape=obs_history.shape[0])
    elif dropout_scheme == 'mnar.v0':
        x, x_dot, theta, theta_dot = obs_history[:, -1, :].T
        terminated = np.logical_or.reduce((x < -x_threshold, x > x_threshold, 
                theta < -theta_threshold_radians, theta > theta_threshold_radians)).astype(bool)
        # logit = 7 - 3 * terminated
        # logit = 5 - 3 * terminated
        # logit = 5 - 2 * (np.power(x, 2) < 1.2)
        # logit = 5 - 2.5 * (x > 0)
        logit = 4 - 6 * np.power(x, 2)
        # logit = 4 - 5 * np.power(x, 2)
        # logit = 5 - 4 * np.power(theta, 2)
        # logit = 5 - 0.5 * np.power(x, 2) - 0.5 * np.power(theta, 2)
        # logit = -5 + 10 * reward_history[:, -1]
        # logit = -4 + 10 * reward_history[:, -1]
        # logit = 4 - 5 * reward_history[:, -1]
        # logit = 4 - 3 * reward_history[:, -1]
        # logit = 6 - 4 * reward_history[:, -1]
        # logit = 6 - 3 * reward_history[:, -1]
        prob = 1 / (np.exp(logit) + 1)
    elif dropout_scheme == 'mar.v0':
        x, x_dot, theta, theta_dot = obs_history[:, -2, :].T
        terminated = np.logical_or.reduce((x < -x_threshold, x > x_threshold, 
            theta < -theta_threshold_radians, theta > theta_threshold_radians)).astype(bool)
        # logit = 7 - 3 * terminated
        # logit = 6 - 4 * reward_history[:, -2]
        logit = 6 - 3 * reward_history[:, -2]
        prob = 1 / (np.exp(logit) + 1)
    else:
        raise NotImplementedError
    return prob.reshape(-1, 1)

class DefaultCartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    Note: This class is directly taken from the gym packeage.

    ### Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ### Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ### Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ### Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ### Arguments

    ```
    gym.make('CartPole-v1')
    ```

    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description
    
    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.
    
    ### Action Space
    
    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.
    
    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |
    
    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
    
    ### Observation Space
    
    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
    
    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |
    
    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)
    
    ### Rewards
    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.
    
    ### Starting State
    
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`
    
    ### Episode End
    
    The episode ends if any one of the following occurs:
    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)
    
    ### Arguments
    T (int): maximum number of steps
    noise_std (float): standard deviation of noise added to the transition

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, 
        render_mode: Optional[str] = "rgb_array", 
        T: int = 100, 
        noise_std: float = 0.02, 
        dropout_scheme: str = '0',
        dropout_rate: float = 0.,
        dropout_obs_count_thres: int = 1,
        # dropout_model: callable = None
    ):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

        # custom attributes
        self.dim = 4
        self.T = T
        self.noise_std = np.repeat(noise_std, 4)
        self.step_count = 0
        self.low = self.observation_space.low
        self.high = self.observation_space.high
        self._np_random = np.random
        # if dropout_model:
        #     assert callable(dropout_model)
        # else:
        #     dropout_model = constant_fn(val=0)  # no dropout
        self.dropout_model = partial(
            dropout_model, 
            T=T, 
            dropout_scheme=dropout_scheme, 
            dropout_rate=dropout_rate,
            dropout_obs_count_thres=dropout_obs_count_thres
        )

    # custom
    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    # custom
    def cal_reward(self, x, theta):
        # x and theta represents cart position and pole angle respectively
        # https://towardsdatascience.com/infinite-steps-cartpole-problem-with-variable-reward-7ad9a0dcf6d0
        reward = (1 - 5 * (x ** 2) / 11.52 - 5 * (theta ** 2) / 288) # r1
        return reward

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        self.step_count += 1 # custom
        self.actions_history.append(action)

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        # self.state = (x, x_dot, theta, theta_dot) # transition without noise
        # custom
        errors = np.random.randn(4) * self.noise_std
        self.state = (x + errors[0], x_dot + errors[1], theta + errors[2], theta_dot + errors[3])

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.step_count >= self.T # custom
        )

        # if not terminated:
        #     reward = 1.0
        # elif self.steps_beyond_terminated is None:
        #     # Pole just fell!
        #     self.steps_beyond_terminated = 0
        #     reward = 1.0
        # else:
        #     if self.steps_beyond_terminated == 0:
        #         logger.warn(
        #             "You are calling 'step()' even though this "
        #             "environment has already returned terminated = True. You "
        #             "should always call 'reset()' once you receive 'terminated = "
        #             "True' -- any further steps are undefined behavior."
        #         )
        #     self.steps_beyond_terminated += 1
        #     reward = 0.0
        
        # custom
        reward = self.cal_reward(x, theta)
        self.rewards_history.append(reward)
        self.states_history = np.concatenate([
            self.states_history, np.array(self.state).reshape(1,-1)], axis=0)
        
        if len(self.rewards_history) < 2:
            self.dropout_prob = self.dropout_model(
                obs_history=self.states_history,
                action_history=self.actions_history,
                reward_history=[0] + self.rewards_history)  # (num_envs,1)
        else:
            self.dropout_prob = self.dropout_model(
                obs_history=self.states_history,
                action_history=self.actions_history,
                reward_history=self.rewards_history)  # (num_envs,1)  
        self.survival_prob = self.next_survival_prob
        self.next_survival_prob *= 1 - self.dropout_prob
        self.dropout_next = (self.dropout_next == 1) * 1 + (
            self.dropout_next < 1) * (self.np_random.uniform(
                low=0, high=1) < self.dropout_prob)  # (num_envs,1)   
        terminated = self.dropout_next or self.step_count >= self.T # custom  
        if self.step_count >= self.T:
            self.step_count = 0
        # if self.render_mode == "human":
        #     self.render()
        env_infos = {
            'next_survival_prob': self.next_survival_prob,
            'dropout': self.dropout_next,
            'dropout_prob': self.dropout_prob
        }
        return np.array(self.state, dtype=np.float32), reward, terminated, env_infos

    def reset(
        self,
        *,
        state: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # super().reset(seed=seed)
        # self.seed(seed=seed)

        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05
        )  # default low and high
        if state is not None:
            self.state = state
        else:
            self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None
        self.step_count = 0 # custom

        # if self.render_mode == "human":
        #     self.render()

        # custom
        self.actions_history = []
        self.rewards_history = []
        self.survival_prob = 1
        self.next_survival_prob = 1
        self.dropout_next = 0
        self.states_history = self.state.reshape(1,-1)
        
        return np.array(self.state, dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class CartPoleVectorEnv(VectorEnv):
    """
    Vectorized CartPole environment that runs multiple environments in parallel.
    
    This is not the same as 1 environment that has multiple subcomponents, but it is many copies of the same base env.
    
    Each observation returned from vectorized environment is a batch of observations for each parallel environment.
    And :meth:`step` is also expected to receive a batch of actions for each parallel environment.
    
    Notes:
        All parallel environments should share the identical observation and action spaces.
        In other words, a vector of multiple different environments is not supported.
    """
    def __init__(
        self, 
        num_envs, 
        T=100, 
        noise_std=0.02, 
        dropout_scheme='0', 
        dropout_rate=0.,
        dropout_obs_count_thres=1
    ):
        """
        Args:
            num_envs (int): number of environments in parallel
            T (int): horizon
        """
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(-high, high, shape=(4, ), dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)

        self.T = T
        self.dim = 4
        self.noise_std = np.tile(noise_std, reps=(4,1))
        self.step_count = 0
        self.low = self.single_observation_space.low
        self.high = self.single_observation_space.high
        self._np_random = np.random
        # if vec_dropout_model:
        #     assert callable(vec_dropout_model)
        # else:
        #     vec_dropout_model = constant_vec_fn(val=0, output_dim=self.num_envs)
        self.dropout_model = partial(
            vec_dropout_model, 
            T=T, 
            dropout_scheme=dropout_scheme, 
            dropout_rate=dropout_rate, 
            dropout_obs_count_thres=dropout_obs_count_thres
        )

        # self.seed()

    # custom
    def cal_reward(self, x, theta):
        # x and theta represents cart position and pole angle respectively
        # https://towardsdatascience.com/infinite-steps-cartpole-problem-with-variable-reward-7ad9a0dcf6d0
        reward = (1 - 5 * (x ** 2) / 11.52 - 5 * (theta ** 2) / 288)
        # reward = (1 - 20 * (theta ** 2) / 288)
        return reward

    def reset_async(self, S_inits=None, seed=None):
        # self.seed(seed=seed)

        self.step_count = 0
        self.actions_history = None  # (num_env,T)
        self.rewards_history = None  # (num_env,T)
        self.survival_prob = np.ones(shape=(self.num_envs, 1),
                                     dtype=np.float32)
        self.next_survival_prob = np.ones(shape=(self.num_envs, 1),
                                          dtype=np.float32)
        self.dropout_next = np.zeros(shape=(self.num_envs, 1), dtype=np.int8)
        self.state_mask = np.ones(shape=(self.num_envs, 1), dtype=np.int8)
        self.next_state_mask = np.ones(shape=(self.num_envs, 1), dtype=np.int8)

        if S_inits is not None:
            assert len(
                S_inits
            ) == self.num_envs, "The length of S_inits should be the same as num_envs"
        else:
            # S_inits = self.observation_space.sample()
            # S_inits = self._np_random.uniform(low=-0.05, high=0.05, size=(self.num_envs, 4))
            S_inits = self._np_random.uniform(low=[-0.5,-1,-2,-3], high=[0.5, 1, 2, 3], size=(self.num_envs, 4))
        # S_inits = np.clip(a=S_inits, a_min=self.observation_space.low, a_max=self.observation_space.high)

        self.steps_beyond_terminated = np.repeat(None, self.num_envs)
        self.states = S_inits  # (num_env,dim)
        self.states_history = np.expand_dims(a=S_inits,
                                             axis=1)  # (num_env,1,dim)
        self.states_history_mask = self.state_mask  # (num_env,1)

    def reset_wait(self):
        """		
        Returns:
            observations (sample from observation_space): a batch of observations from the vectorized environment.
		"""
        return self.states

    def reset(self, S_inits=None, seed=None):
        """Reset all sub-environments and return a batch of initial observations.
        
        Returns:
            observations (sample from observation_space): a batch of observations from the vectorized environment.
        """
        self.reset_async(S_inits=S_inits, seed=seed)
        return self.reset_wait()

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def step_async(self, actions):
        """Asynchronously performs steps in the sub-environments.
       
        The results can be retrieved via a call to :meth:`step_wait`.
       
        Args:
            actions: The actions to take asynchronously
        """
        self.step_count += 1
        
        if isinstance(actions, np.ndarray):
            actions = list(actions.squeeze())

        assert self.single_action_space.contains(actions[0])
        assert self.states is not None, "Call reset before using step method."
        actions_arr = np.array(actions).reshape(-1, 1)
        if self.actions_history is None:
            self.actions_history = actions_arr
        else:
            self.actions_history = np.concatenate(
                [self.actions_history, actions_arr], axis=1)
        
        x, x_dot, theta, theta_dot = self.states.T # (num_envs, 4) -> (4, num_envs)
        force = np.array([self.force_mag if action == 1 else -self.force_mag for action in actions])
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        # next state
        errors = np.random.randn(4, self.num_envs) * self.noise_std
        next_states = np.array([x + errors[0], x_dot + errors[1], theta + errors[2], theta_dot + errors[3]]).T # (num_envs, 4)
        if next_states.ndim == 1:
            next_states = next_states.reshape(1, -1)
        
        # reward
        rewards = self.cal_reward(x, theta).reshape(-1,1)  # (num_envs,1)

        if self.step_count <= self.T: # <= instead of <
            self.states_history = np.concatenate(
                [self.states_history,
                 np.expand_dims(a=next_states, axis=1)],
                axis=1)
        if self.rewards_history is None:
            self.rewards_history = rewards
        else:
            self.rewards_history = np.concatenate(
                [self.rewards_history, rewards], axis=1)
        
        # dropout_prob
        if self.rewards_history.shape[1] < 2:
            self.dropout_prob = self.dropout_model(
                obs_history=self.states_history,
                action_history=self.actions_history,
                reward_history=np.concatenate(
                    [np.zeros(shape=(self.num_envs, 1)), rewards],
                    axis=1))  # (num_envs,1)
        else:
            self.dropout_prob = self.dropout_model(
                obs_history=self.states_history,
                action_history=self.actions_history,
                reward_history=self.rewards_history)  # (num_envs,1)
        if self.dropout_prob.ndim == 1:
            self.dropout_prob = self.dropout_prob.reshape(-1,1)
        self.survival_prob = self.next_survival_prob
        self.next_survival_prob *= 1 - self.dropout_prob
        self.dropout_next = (self.dropout_next == 1) * 1 + (
            self.dropout_next < 1) * (self.np_random.uniform(
                low=0, high=1, size=(len(self.dropout_prob), 1)) <
                                      self.dropout_prob)  # (num_envs,1)
        self.next_state_mask = np.minimum(
            self.state_mask,
            1 - self.dropout_next)  # (num_envs,1), element-wise minimum

        if self.step_count <= self.T: # <= instead of <
            self.states_history_mask = np.concatenate(
                [self.states_history_mask, self.next_state_mask],
                axis=1)

        self.dones = np.logical_or.reduce((x < -self.x_threshold
                , x > self.x_threshold, theta < -self.theta_threshold_radians
                , theta > self.theta_threshold_radians
                , np.repeat(self.step_count, self.num_envs) >= self.T)).astype(bool)
        if self.step_count >= self.T:
            self.step_count = 0

        self.state_mask = self.next_state_mask
        self.states = next_states
        self.rewards = rewards


    def step_wait(self):
        """		
		Returns:
		    observations (sample from observation_space): a batch of observations from the vectorized environment.
		    rewards (np.ndarray): a vector of rewards from the vectorized environment.
		    dones (np.ndarray): a vector whose entries indicate whether the episode has ended.
		    infos (list of dict): a list of auxiliary diagnostic information.
		"""
        env_infos = {
            'next_survival_prob': self.next_survival_prob.copy(
            ),  # probability of observing the next step
            'dropout': self.dropout_next.astype(
                np.int8).copy(),  # dropout indicator of the next step 
            'dropout_prob': self.dropout_prob.copy(
            ),  # dropout probability of the current step
            'state_mask':
            self.state_mask.copy()  # 1 indicates observed, 0 otherwise
        }

        return self.states.copy(), self.rewards.copy(), self.dones.copy(
            ), env_infos  # create a copy to aviod mutating values

    def step(self, actions):
        """Take an action for each parallel environment.
        Args:
            actions: element of :attr:`action_space` Batch of actions.
        Returns:
            Batch of (observations, rewards, terminated, truncated, infos) or (observations, rewards, dones, infos)
        """
        self.step_async(actions)
        return self.step_wait()


if __name__ == '__main__':
    # env = CartPoleEnv(
    #     T=100, 
    #     noise_std=0.1, 
    #     dropout_scheme='mar.v0', 
    #     dropout_rate=0.9, 
    #     dropout_obs_count_thres=10)
    # _ = env.reset(seed=0)
    # done = False
    # step = 0
    # while not done:
    #     _, _, done, _ = env.step(action=np.random.randint(2))
    #     step += 1
    # print('num steps:', step)

    num_envs = 10
    vec_env = CartPoleVectorEnv(
        num_envs=num_envs, 
        T=100, 
        noise_std=0.1, 
        dropout_scheme='mnar.v0', 
        dropout_rate=0.9, 
        dropout_obs_count_thres=10
    )
    _ = vec_env.reset(seed=0)
    max_dropout_prob = 0
    for _ in range(200):
        _, _ , _, env_infos = vec_env.step(actions=np.random.randint(low=0, high=2,size=(num_envs,1)))
        max_dropout_prob = max(max_dropout_prob, np.max(env_infos['dropout_prob']))
    print('missing rate:', 1 - np.mean(vec_env.states_history_mask[:, -1]))
    print('max dropout prob:', max_dropout_prob)