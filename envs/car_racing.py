import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
import math
import pyglet
from pyglet import shapes

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
        max_laps: int = 1,
    ):
        super(CarRacing, self).__init__()
        self._env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=continuous, lap_complete_percent=lap_complete_percent)
        self._env = TimeLimit(self._env, max_episode_steps=max_episode_steps)

        self.render_mode = render_mode
        self.continuous = continuous
        self.lap_complete_percent = lap_complete_percent
        self._reward_shaping = reward_shaping

        self.max_laps = max(1, int(max_laps))
        self._current_lap = 1

        # State S_t
        image_space = spaces.Box(0, 255, shape=(96, 96, 3), dtype=np.uint8)

        # state branch: [d_t, v_t, infield, pitroad, ell_t, w_t, f_t, kappa_t]
        state_low  = np.array([-5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05], dtype=np.float32)
        state_high = np.array([ 5.0, 70.0, 1.0, 1.0, 1.0, 1.0, 1.0,  0.05], dtype=np.float32)
        state_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)

        self.observation_space = spaces.Dict({"image": image_space, "state": state_space})

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

        self._gauge = None # the gauge window

    # reset()
    def reset(self, *, seed=None, options=None):
        observation, info = self._env.reset(seed=seed, options=options)

        self.progress = 0.0
        self._last_progress = 0.0

        self._wear = 0.0
        self._fuel = 1.0

        self._current_lap = 1

        info["lap"] = self._current_lap
        info["max_laps"] = self.max_laps

        return self._get_obs(observation), info
    
    # step()
    def step(self, action):
        # Map our action to base env (3 controls) + pit command
        base_action, pit_command = self._map_action(action)
        
        # Step in base environment
        observation, reward, terminated, truncated, info = self._env.step(base_action)

        # Get speed from box2d
        velocity = self._get_velocity()
        # print(f"Velocity: {velocity:.2f} | Fuel: {self._fuel:.3f} | Wear: {self._wear:.3f}")

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

        # per-lap progress + overall progress
        base_env = self._env.unwrapped
        if getattr(base_env, "track", None):
            ell_t = base_env.tile_visited_count / len(base_env.track)
        else:
            ell_t = 0.0

        # overall race progress in [0,1]
        self.progress = ((self._current_lap - 1) + ell_t) / float(self.max_laps)

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

        env_done = bool(terminated or truncated)
        lap_finished = env_done and (ell_t >= self.lap_complete_percent - 1e-6)

        info["lap"] = self._current_lap
        info["max_laps"] = self.max_laps
        info["lap_finished"] = False

        if lap_finished and self._current_lap < self.max_laps:
            # Completed a lap, but not the whole race: start a new lap.
            info["lap_finished"] = True
            finished_lap = self._current_lap
            self._current_lap += 1

            # Reset underlying env to start next lap (same map)
            observation, info_reset = self._env.reset()

            # Do not end the episode 
            terminated = False
            truncated = False

            info["finished_lap"] = finished_lap
            info["lap"] = self._current_lap
            info["reset_for_new_lap"] = True
        else:
            info["reset_for_new_lap"] = False

        # Update observation
        return self._get_obs(observation), reward, terminated, truncated, info
    
    # render()
    def render(self):
        if self.render_mode == "rgb_array":
            return self._env.render()
        out = self._env.render()

        # Update or create the separate gauge window
        try:
            self._ensure_gauge_window()
            self._gauge.update(fuel=self._fuel, tire_health=1.0 - self._wear)
        except Exception:
            pass

        return out

    def close(self):
        try:
            if self._gauge is not None:
                self._gauge.close()
                self._gauge = None
        except Exception:
            pass
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
        v_t = self._get_velocity()
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
    
    def _compute_offset(self):
        env = self._env.unwrapped

        # Car position in Box2D world coordinates
        car_position = env.car.hull.position
        car_x, car_y = float(car_position[0]), float(car_position[1])

        track = env.track
        if not track or len(track) == 0:
            return 0.0

        # Find the track tile whose center (x, y) is closest to the car
        min_distance_squared = float("inf")
        nearest_tile_index = 0
        for i, (tile_alpha, tile_beta, tile_center_x, tile_center_y) in enumerate(track): 
            delta_x = car_x - tile_center_x
            delta_y = car_y - tile_center_y
            distance_squared = delta_x * delta_x + delta_y * delta_y
            if distance_squared < min_distance_squared:
                min_distance_squared = distance_squared
                nearest_tile_index = i

        # Get the center and orientation (beta) of that nearest tile
        _, nearest_tile_beta, nearest_tile_center_x, nearest_tile_center_y = track[nearest_tile_index]

        # Compute vector from centerline point to car
        vector_to_car_x, vector_to_car_y = car_x - nearest_tile_center_x, car_y - nearest_tile_center_y

        # Road normal is (cos(beta), sin(beta)) as used in gym's create_track()
        # It is a vector that points sideways, perpendicular to the direction you are driving
        road_normal_x, road_normal_y = math.cos(nearest_tile_beta), math.sin(nearest_tile_beta)

        # Signed lateral offset = projection of w onto the normal, 
        signed_distance = (vector_to_car_x * road_normal_x + vector_to_car_y * road_normal_y)

        return float(np.clip(signed_distance, -5.0, 5.0))

    def _compute_lookahead_curvature(self):
        # Placeholder implementation
        return 0.0 # kappa_t (curvature)

    def _is_infield_from_offset(self):
        # Placeholder implementation
        return False

    def _is_in_pit(self):
        # Placeholder implementation
        return False

    def _get_velocity(self):
        v_t = self._env.unwrapped.car.hull.linearVelocity
        return np.linalg.norm(v_t)
    
    def _ensure_gauge_window(self):
        if self._gauge is not None:
            return
        try:
            self._gauge = _GaugeWindow(title="Fuel & Tire Gauges", width=280, height=110)
        except Exception:
            self._gauge = None

class _GaugeWindow:
    def __init__(self, title="Fuel & Tire Gauges", width=280, height=110):
        self._pyglet = pyglet
        self.window = pyglet.window.Window(width=width, height=height, caption=title, resizable=False, vsync=False)
        self.batch = pyglet.graphics.Batch()
        self.shapes = {}
        self.labels = {}

        # Layout
        self.margin = 30
        self.bar_w = width - self.margin*2
        self.bar_h = 16
        self.gap = 30

        # Colors
        self.col_bg   = (25, 25, 25)
        self.col_ok   = (80, 200, 120)
        self.col_warn = (240, 200, 80)
        self.col_crit = (230, 80, 80)
        self.col_text = (240, 240, 240, 255)

        # Draws Fuel bar at the top and Tire Wear bar below
        self._make_row("FUEL", y_top=height - self.margin)
        self._make_row("TIRE WEAR", y_top=height - self.margin - (self.bar_h + self.gap))

        # initial draw
        self.update(1.0, 1.0)

    def _make_row(self, name, y_top):
        x0 = self.margin
        self.shapes[(name, "bg")]   = shapes.Rectangle(x0, y_top - self.bar_h, self.bar_w, self.bar_h, color=self.col_bg, batch=self.batch)
        self.shapes[(name, "fill")] = shapes.Rectangle(x0, y_top - self.bar_h, 1, self.bar_h, color=self.col_ok, batch=self.batch)
        self.labels[(name, "left")] = self._pyglet.text.Label(
            name, font_size=10, color=self.col_text, x=x0, y=y_top + 2,
            anchor_x="left", anchor_y="baseline", batch=self.batch
        )
        self.labels[(name, "right")] = self._pyglet.text.Label(
            "100%", font_size=10, color=self.col_text, x=x0 + self.bar_w, y=y_top - self.bar_h + 2,
            anchor_x="right", anchor_y="baseline", batch=self.batch
        )

    def _colour_format(self, v):
        if v > 0.5: return self.col_ok
        if v > 0.2: return self.col_warn
        return self.col_crit

    def update(self, fuel, tire_health):
        pyglet = self._pyglet

        # Keep window responsive
        pyglet.clock.tick()
        self.window.switch_to()
        self.window.dispatch_events()

        fuel = float(np.clip(fuel, 0.0, 1.0))
        tire = float(np.clip(tire_health, 0.0, 1.0))

        # Update fills + labels
        f_fill = self.shapes[("FUEL", "fill")]
        f_fill.width = int(self.bar_w * fuel)
        f_fill.color = self._colour_format(fuel)

        t_fill = self.shapes[("TIRE WEAR", "fill")]
        t_fill.width = int(self.bar_w * tire)
        t_fill.color = self._colour_format(tire)

        self.labels[("FUEL", "right")].text = f"{int(fuel*100)}%"
        self.labels[("TIRE WEAR", "right")].text = f"{int(tire*100)}%"

        # Draw this frame
        self.window.clear()
        self.batch.draw()
        self.window.flip()

    def close(self):
        try:
            self.window.close()
        except Exception:
            pass
    
    # TODO: Implement pitstop rendering

    # TODO: Implement a controller
