import gymnasium as gym
import numpy as np
from gymnasium import spaces

class EnvWrapper(gym.Wrapper):
    # small knobs
    TOTAL_TILES = 1000
    WEAR_RATE_FUEL = 0.0015
    WEAR_RATE_TIRE = 0.0015
    STEP_BONUS = 0.05
    PIT_PENALTY = -2.0
    PIT_COOLDOWN = 25

    # no-progress shaping
    NOPROG_PENALTY = 0.02     # per step with dt==0
    NOPROG_PENALTY_CAP = 0.20 # cap total penalty per step

    def __init__(self, render_mode=None):
        env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=False)
        super().__init__(env)

        # Base env Discrete(5); add action 5 for pit
        self.action_space = spaces.Discrete(6)
        self.observation_space = self.env.observation_space

        self.fuel = 1.0
        self.tire = 1.0
        self.total_tiles = self.TOTAL_TILES
        self.tiles_visited = 0
        self._pit_cd = 0
        self._last_obs = None
        self._no_prog_steps = 0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.fuel = 1.0
        self.tire = 1.0
        self.tiles_visited = 0
        self._pit_cd = 0
        self._last_obs = obs
        self._no_prog_steps = 0
        return obs, info
    
    def step(self, action):
        terminated, truncated = False, False
        pit = False
        pit_blocked = False

        # if pit during cooldown, fall back to ACCELERATE so we don't stall
        if action == 5 and self._pit_cd > 0:
            pit_blocked = True
            action = 3  # accelerate

        if action == 5 and self._pit_cd == 0:
            # pit transition
            pit = True
            self.fuel = 1.0
            self.tire = 1.0
            self._pit_cd = self.PIT_COOLDOWN
            obs = self._last_obs  # keep last frame (avoid zeroing)
            base_reward = 0.0
            info = {"pit_stop": True}
        else:
            # normal step
            obs, base_reward, terminated, truncated, info = self.env.step(action)
            self._last_obs = obs
            self.fuel = max(0.0, self.fuel - self.WEAR_RATE_FUEL)
            self.tire = max(0.0, self.tire - self.WEAR_RATE_TIRE)

        if self._pit_cd > 0:
            self._pit_cd -= 1

        # reward shaping
        delta_tiles = info.get("tile_visited", 0)  # 0 default prevents phantom progress
        self.tiles_visited += delta_tiles

        reward = float(base_reward)
        progress = (1000.0 / self.total_tiles) * float(delta_tiles)

        # mild incentive for healthy resources + keep-moving bonus
        reward += 0.5 * self.fuel + 0.5 * self.tire + self.STEP_BONUS + progress

        # gentle penalty if we're not making progress (e.g., off track)
        if delta_tiles == 0:
            self._no_prog_steps += 1
            reward -= min(self.NOPROG_PENALTY * self._no_prog_steps, self.NOPROG_PENALTY_CAP)
        else:
            self._no_prog_steps = 0

        if info.get("lap_complete", False):
            reward += 100.0
        if pit:
            reward += self.PIT_PENALTY

        info.update({
            "fuel": self.fuel,
            "tire": self.tire,
            "pit_stop": pit,
            "pit_blocked": pit_blocked,
            "pit_cooldown": self._pit_cd,
            "no_progress_steps": self._no_prog_steps
        })

        print(f"fuel={self.fuel:.3f}, tire={self.tire:.3f}, pit={pit}, r={reward:.2f}, dt={delta_tiles}, cd={self._pit_cd}, nps={self._no_prog_steps}")

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

# 0: nothing, 1: left, 2: right, 3: accelerate, 4: brake, 5: pit
def heuristic_action(obs, prev_action=3):
    """
    Keep the gray road centered using a single scanline.
    - Convert to grayscale.
    - Look at a band ahead of the car (rows ~60:75).
    - Compute column of minimum 'greenness' (road is gray; grass is green).
    - Steer left/right if the road center drifts; otherwise accelerate.
    """
    arr = obs
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    h, w, _ = arr.shape
    band = arr[int(0.62*h):int(0.78*h), :, :]    # a horizontal band ahead
    R = band[..., 0].astype(np.float32)
    G = band[..., 1].astype(np.float32)
    B = band[..., 2].astype(np.float32)

    # "how green" each column is, road is gray (lower green excess)
    green_excess = G - 0.5*(R+B)
    col_scores = green_excess.mean(axis=0)

    # road center ~ min green_excess (less green than grass)
    road_center = int(np.argmin(col_scores))
    center = w // 2
    offset = road_center - center

    # simple deadband & smoothing
    DEAD = 4
    if offset < -DEAD:
        steer = 1  # left
    elif offset > DEAD:
        steer = 2  # right
    else:
        steer = 3  # go straight/accelerate

    # optionally keep braking if we were braking last frame
    if prev_action == 4 and steer == 3:
        return 4  # short brake persistence
    return steer

if __name__ == "__main__":
    USE_HEURISTIC = True   # flip to False to see random policy

    env = EnvWrapper(render_mode="human")
    obs, info = env.reset()
    done = False
    ep_reward = 0.0
    prev = 3

    while not done:
        if USE_HEURISTIC:
            action = heuristic_action(obs, prev)
            # add a little throttle persistence
            if action == 3 and np.random.rand() < 0.1:
                action = 3
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        done = terminated or truncated
        prev = action

    print(f"Reward: {ep_reward:.2f}")
    env.close()
