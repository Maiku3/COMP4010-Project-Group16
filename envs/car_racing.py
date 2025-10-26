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
        image_space = spaces.Box(0, 255, shape=(96, 96, 3), dtype=np.uint8)

        # state branch: [d_t, v_t, infield, pitroad, ell_t, w_t, f_t, kappa_t, progress]
        state_low  = np.array([-5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05, 0.0], dtype=np.float32)
        state_high = np.array([ 5.0, 70.0, 1.0, 1.0, 1.0, 1.0, 1.0,  0.05, 1.0], dtype=np.float32)
        state_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)

        self.observation_space = spaces.Dict({"image": image_space, "state": state_space})

        # self.observation_space = spaces.Dict({
        #             "image": spaces.Box(0, 255, shape=(96, 96, 3), dtype=np.uint8),
        #             "d_t": spaces.Box(-5.0, 5.0, shape=(1,), dtype=np.float32),
        #             "v_t": spaces.Box(0.0, 70.0, shape=(1,), dtype=np.float32),
        #             "infield": spaces.Discrete(2),
        #             "pitroad": spaces.Discrete(2),
        #             "ell_t": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        #             "w_t": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        #             "f_t": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        #             "kappa_t": spaces.Box(-0.05, 0.05, shape=(1,), dtype=np.float32),
        #             "progress": spaces.Box(np.array([0.0], np.float32), np.array([1.0], np.float32)),
        #         })

        # Action A_t
        if self.continuous:
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                shape=(4,),
                dtype=np.float32,
            )
        else:
            # 0:nothing, 1:steer right, 2:steer left, 3:gas, 4:brake, 5:pit
            self.action_space = spaces.Discrete(6)

        # Internal states
        self._wear = 0.0   # 0=new, 1=fully_worn
        self._fuel = 1.0   # 1=full, 0=empty
        self._dt = 1.0 / 50.0  # CarRacing runs at 50 frames per second

        # Fuel (per second) 
        # At idle: drains slowly, but at full gas: drains more (will need to adjust)
        self.fuel_base_per_s = 0.0015         # idle consumption
        self.fuel_full_per_s = 0.0110         # extra at gas=1 (so total ~0.0125/s at speed)

        # Wear (per second)
        self.wear_base_per_s = 0.0003         # always-on tiny wear
        self.wear_brake_per_s = 0.0030        # additional at brake=1
        self.wear_steer_per_s = 0.0020        # additional at steer=1

        # Speed scaling reference (m/s); above this speed, scaling is 1.0
        self.speed_ref_mps = 40.0 # Will need to adjust based on testing

        self.progress = 0.0 # overall progress in [0,1] if we want to track
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
        # Map our action to base env (3 controls) + pit command
        base_action, pit_command = self._map_action(action)
        
        # Step in base environment
        observation, reward, terminated, truncated, info = self._env.step(base_action)

        # Get speed from box2d
        velocity = np.linalg.norm(self._env.unwrapped.car.hull.linearVelocity)
        print(f"Velocity: {velocity:.2f} | Fuel: {self._fuel:.3f} | Wear: {self._wear:.3f}")

        # Update resources
        steer, gas, brake = float(base_action[0]), float(base_action[1]), float(base_action[2])
        
        # Scale effects by speed (0.3 at standstill to 1.0 at speed_ref and above)
        speed_scale = 0.3 + 0.7 * min(1.0, velocity / self.speed_ref_mps)

        # Fuel: from idle to full-throttle rate, then scale by speed.
        fuel_rate_per_s = self.fuel_base_per_s + gas * (self.fuel_full_per_s - self.fuel_base_per_s)
        fuel_rate_per_s *= speed_scale
        self._fuel = max(0.0, self._fuel - self._dt * fuel_rate_per_s)

        # Wear: base + brake + steering contributions, then scale by speed.
        wear_rate_per_s = (
            self.wear_base_per_s
            + brake * self.wear_brake_per_s
            + abs(steer) * self.wear_steer_per_s
        )
        wear_rate_per_s *= speed_scale
        self._wear = min(1.0, self._wear + self._dt * wear_rate_per_s)

        # Placeholder for infield and pitroad detection
        infield = False
        pitroad = False

        # Pit Stop Logic
        pit_executed = False
        if pit_command and infield and pitroad and (velocity < 5.0):
            self._fuel = 1.0
            self._wear = 0.0
            pit_executed = True
        info["pit_executed"] = pit_executed

        # Update observation
        return self._get_obs(observation), reward, terminated, truncated, info
    
    # render()
    def render(self):
        return self._env.render()
    
    def close(self):
        self._env.close()

    # Helper Functions
    def _map_action(self, action):
        # Base CarRacing expects only 3 controls, so we keep 'pit' separate
        if self.continuous:
            a = np.asarray(action, dtype=np.float32)
            steer = float(np.clip(a[0], -1.0, 1.0))
            gas   = float(np.clip(a[1],  0.0, 1.0))
            brake = float(np.clip(a[2],  0.0, 1.0))
            pit   = bool(a[3] >= 0.5)
            return np.array([steer, gas, brake], dtype=np.float32), pit
        else:
            a = int(action)
            pit = (a == 5)
            if a == 0:   base = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            elif a == 1: base = np.array([+1.0, 0.0, 0.0], dtype=np.float32)
            elif a == 2: base = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
            elif a == 3: base = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            elif a == 4: base = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            elif a == 5: base = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else: raise ValueError(f"Invalid discrete action {a}")
            return base, pit

    def _get_obs(self, base_obs):
        d_t = self._compute_offset()
        v_t = np.linalg.norm(self._env.unwrapped.car.hull.linearVelocity)
        ell_t = self._env.unwrapped.tile_visited_count / len(self._env.unwrapped.track)

        infield = self._is_infield_from_offset()
        pitroad = self._is_in_pit()

        kappa_t = self._compute_lookahead_curvature()

        state_vec = np.array([
            float(np.clip(d_t, -5.0, 5.0)), # d_t: lateral distance from track center
            float(np.clip(v_t, 0.0, 70.0)), # v_t: current velocity
            1.0 if infield else 0.0, # infield: whether the car is in the infield
            1.0 if pitroad else 0.0, # pitroad: whether the car is on the pit road
            float(np.clip(ell_t, 0.0, 1.0)), # ell_t: progress along the track
            float(np.clip(self._wear, 0.0, 1.0)), # w_t: tire wear
            float(np.clip(self._fuel, 0.0, 1.0)), # f_t: fuel level
            float(np.clip(kappa_t, -0.05, 0.05)), # kappa_t: track curvature
        ], dtype=np.float32)

        return {"image": base_obs, "state": state_vec}
        # return {
        #     "image": base_obs,
        #     "d_t": np.array([0.0], dtype=np.float32),
        #     "v_t": np.array([0.0], dtype=np.float32),
        #     "infield": 0,
        #     "pitroad": 0,
        #     "ell_t": np.array([0.0], dtype=np.float32),
        #     "w_t": np.array([self._wear], dtype=np.float32),
        #     "f_t": np.array([self._fuel], dtype=np.float32),
        #     "kappa_t": np.array([0.0], dtype=np.float32),
        #     "progress": np.array([self.progress], dtype=np.float32),
        # }
    
    def _compute_offset(self):
        # Placeholder implementation
        # geometry calculations based on the car's position relative to the track's centerline.
        # For now, return dummy values
        return 0.0  # d_t (offset)

    def _compute_lookahead_curvature(self):
        # Placeholder implementation
        return 0.0 # kappa_t (curvature)

    def _is_infield_from_offset(self):
        # Placeholder implementation
        return False

    def _is_in_pit(self):
        # Placeholder implementation
        return False

    
    # TODO: Implement pitstop rendering

    # TODO: Implement a controller