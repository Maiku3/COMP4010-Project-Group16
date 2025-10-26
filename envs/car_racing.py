import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

class CarRacing(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # __init__()
    def __init__(
        self,
        render_mode: str | None = None,
        continuous: bool = True,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        reward_shaping: bool = True,
        max_episode_steps: int = 3000,
    ):
        super(CarRacing, self).__init__()
        self._env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=continuous, lap_complete_percent=lap_complete_percent)
        self._env = TimeLimit(self._env, max_episode_steps=max_episode_steps)

        self.render_mode = render_mode
        self.continuous = continuous
        self.lap_complete_percent = lap_complete_percent
        self._reward_shaping = reward_shaping

        # State S_t
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(96, 96, 3), dtype=np.uint8),
            "d_t": spaces.Box(-5.0, 5.0, shape=(1,), dtype=np.float32),
            "v_t": spaces.Box(0.0, 70.0, shape=(1,), dtype=np.float32),
            "infield": spaces.Discrete(2),
            "pitroad": spaces.Discrete(2),
            "ell_t": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "w_t": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "f_t": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "kappa_t": spaces.Box(-0.05, 0.05, shape=(1,), dtype=np.float32),
            "progress": spaces.Box(np.array([0.0], np.float32), np.array([1.0], np.float32)),
        })

        # Action A_t
        if self.continuous:
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                shape=(4,),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Discrete(5)

        self._wear = 0.0   # 0=new, 1=fully_worn
        self._fuel = 1.0   # 1=full, 0=empty

        self._wear_rate_base = 0.0001
        self._fuel_rate_base = 0.0001

        self.progress = 0.0
        self._last_progress = 0.0

    # reset()
    def reset(self, *, seed=None, options=None):
        observation, info = self._env.reset(seed=seed, options=options)

        self.progress = 0.0
        self._last_progress = 0.0

        self._wear = 0.0
        self._fuel = 1.0

        return self._get_obs(observation), info
    
    # step()
    def step(self, action):
        observation, reward, terminated, truncated, info = self._env.step(action)

        self._wear += self._wear_rate_base * (1 - self._wear)
        self._fuel -= self._fuel_rate_base * (1 - self._fuel)

        # Pit Stop
        if info.get("pitroad", 0) == 1 and info.get("v_t", 0) < 5.0:
            if self.continuous:
                pit_command = action[3] > 0.5
            else:
                pit_command = action == 5

            if pit_command:
                self._wear = 0.0
                self._fuel = 1.0
                pit_executed = True
            else:
                pit_executed = False
        else:
            pit_executed = False
        info["pit_executed"] = pit_executed

        return self._get_obs(observation), reward, terminated, truncated, info
    
    # render()
    def render(self):
        return self._env.render()
    
    def close(self):
        self._env.close()

    def _get_obs(self, base_obs):
        return {
            "image": base_obs,
            "d_t": np.array([0.0], dtype=np.float32),
            "v_t": np.array([0.0], dtype=np.float32),
            "infield": 0,
            "pitroad": 0,
            "ell_t": np.array([0.0], dtype=np.float32),
            "w_t": np.array([self._wear], dtype=np.float32),
            "f_t": np.array([self._fuel], dtype=np.float32),
            "kappa_t": np.array([0.0], dtype=np.float32),
            "progress": np.array([self.progress], dtype=np.float32),
        }
    
    # TODO: Implement a controller
