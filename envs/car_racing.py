import gymnasium as gym
import numpy as np
from gymnasium import spaces

class EnvWrapper(gym.Wrapper):

    def __init__(self, render_mode=None):
        env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=False)
        super().__init__(env)

        # Action space
        # Base Env: Discrete(5) 
        # add action 5 for pit
        self.action_space = spaces.Discrete(6)

        # State space
        self.observation_space = self.env.observation_space

        self.fuel = 1.0
        self.tire = 1.0
        self.total_tiles = 1000
        self.tiles_visited = 0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.fuel = 1.0
        self.tire = 1.0
        self.tiles_visited = 0
        return obs, info
    
    def step(self, action):
        terminated, truncated = False, False
        pit = False
        step_bonus = 0.0

        # Transition dynamic
        if action == 5:  # pit
            pit = True
            self.fuel = 1.0
            self.tire = 1.0
            reward = 5.0  # small reward to use pit 
            obs = np.zeros_like(self.env.observation_space.sample())
            info = {"pit_stop": True}
        else:
            # normal driving
            obs, base_reward, terminated, truncated, info = self.env.step(action)
            
            self.fuel = max(0.0, self.fuel - 0.001)
            self.tire = max(0.0, self.tire - 0.001)
            self.tiles_visited += info.get("tile_visited", 1)

            # Reward structure
            # Base reward from Gym
            reward = max(0.0, base_reward)

            progres = (1000 / self.total_tiles) * info.get("tile_visited", 1)

            # reward for good condition
            fuel = self.fuel * 0.5
            tire = self.tire * 0.5

            # small reward for keep moving
            step_bonus = 0.1

            reward = reward + progres + fuel + tire + step_bonus

            if info.get("lap_complete", False):
                reward += 100.0  # Reward for finishing lap

        info.update({
            "fuel": self.fuel,
            "tire": self.tire,
            "pit_stop": pit
        })
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

if __name__ == "__main__":
    env = EnvWrapper(render_mode="human")
    obs, info = env.reset()
    done = False
    ep_reward = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        done = terminated or truncated

    print(f"Reward: {ep_reward:.2f}")
    env.close()
