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
        base_env = self._env.unwrapped
        self._env = TimeLimit(base_env, max_episode_steps=max_episode_steps)

        self.render_mode = render_mode
        self.continuous = continuous
        self.lap_complete_percent = lap_complete_percent
        self._reward_shaping = reward_shaping

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

        # ===== Fuel (per second) ====== 
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

        # ==== Pitstop configuration ====
        self._pit_center_test = False       # toggle to True to center pit stops and test
        self._pit_speed_max = 5.0           # must be slow to service
        self._pit_count = 25                # number of pit sectors per lap
        self._pit_sector_width = 0.0075     # sector width in progress units

        # Painted strip geometry (world-space)
        self._pit_sign = 1.0
        self._pit_inner_offset = 3.2        # start position relative to centerline
        self._pit_width = 3.5               # thickness of the painted strip

        # Detection must match the paint exactly:
        self._pit_polys_count = 0
        self._prev_pitroad     = False
        self._pit_lock_sector  = None        # prevents spamming within the current sector

        self._sync_pit_bounds()
        # ===============================

        # ==== How much wear impacts steering and gas ====
        self.steering_grip_min = 0.3      # min grip at wear=1.0 (30% of steering)
        self.steering_wear_strength = 0.0 # 1.0 = full effect, < 1.0 = weaker effect

        # Multi-lap state
        self.max_laps = max(1, int(max_laps))
        self._current_lap = 1

    # reset()
    def reset(self, *, seed=None, options=None):
        observation, info = self._env.reset(seed=seed, options=options)

        self.progress = 0.0
        self._last_progress = 0.0

        self._wear = 0.0
        self._fuel = 1.0

        self._prev_pitroad = False

        # reset lap counter
        self._current_lap = 1
        info["lap"] = self._current_lap
        info["max_laps"] = self.max_laps

        # build static painted pit strips onto the track
        try:
            self._sync_pit_bounds() 
            self._build_pit_polys()
        except Exception:
            pass

        return self._get_obs(observation), info
    
    # step()
    def step(self, action):
        # Map our action to base env (3 controls) + pit command
        base_action, pit_command = self._map_action(action)
        
        # Get actions
        steer, gas, brake = base_action[0], base_action[1], base_action[2]

        # Step in base environment
        observation, reward, terminated, truncated, info = self._env.step(base_action)

        # If the car is off the track surface (and not in the pit),
        # immediately terminate the episode.
        if self._is_infield_from_offset() and not self._is_in_pit():
            terminated = True
            # Optional strong penalty to discourage leaving the track
            reward -= 20.0

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

        # per-lap progress on the current track
        ell_t = self._env.unwrapped.tile_visited_count / len(self._env.unwrapped.track)

        env_done = bool(terminated or truncated)
        lap_finished = env_done and (ell_t >= self.lap_complete_percent - 1e-6)

        info["lap"] = self._current_lap
        info["max_laps"] = self.max_laps
        info["lap_finished"] = False
        info["reset_for_new_lap"] = False

        if lap_finished and self._current_lap < self.max_laps:
            # Finished a lap but not the whole race: start a new lap
            finished_lap = self._current_lap
            self._current_lap += 1
            info["lap_finished"] = True
            info["finished_lap"] = finished_lap

            # Reset underlying env to start the next lap
            observation, info_reset = self._env.reset()

            # Keep fuel/wear across laps (endurance), but reset some internal race state
            self.progress = 0.0
            self._last_progress = 0.0
            self._prev_pitroad = False
            self._pit_lock_sector = None

            try:
                self._sync_pit_bounds()
                self._build_pit_polys()
            except Exception:
                pass

            # Don't end the episode from the agent's POV
            terminated = False
            truncated = False

            # Recompute progress on the fresh lap
            ell_t = self._env.unwrapped.tile_visited_count / len(self._env.unwrapped.track)
            info["reset_for_new_lap"] = True
        else:
            # Race progress across all laps [0,1]
            if self.max_laps > 0:
                self.progress = ((self._current_lap - 1) + ell_t) / float(self.max_laps)
        d_t = self._compute_offset()

        tile_idx = self._nearest_tile_index()   # tile index drives sector logic
        in_sector, sector_idx = self._sector_state_by_index(tile_idx)
        in_strip = (self._pit_d_min <= d_t <= self._pit_d_max)
        pitroad = bool(in_sector and in_strip)

        # edge logs
        pit_enter = pitroad and not self._prev_pitroad
        pit_exit  = (not pitroad) and self._prev_pitroad
        self._prev_pitroad = pitroad

        # clear lock when we're not in any sector (so next sector can service)
        if not in_sector:
            self._pit_lock_sector = None

        pit_executed = False
        if pit_enter and (velocity < self._pit_speed_max):
            # only service if we haven't serviced this sector yet
            if self._pit_lock_sector is None or self._pit_lock_sector != sector_idx:
                self._pit_lock_sector = sector_idx
                self._fuel = 1.0
                self._wear = 0.0
                pit_executed = True
                print(f"[PIT] SERVICE at sector={sector_idx} ell={ell_t:.3f} d_t={d_t:+.2f}")

        # logs to check edges
        if pit_enter:
            print(f"[PIT] ENTER sector={sector_idx} ell={ell_t:.3f} d_t={d_t:+.2f}")
        if pit_exit:
            print(f"[PIT] EXIT  ell={ell_t:.3f} d_t={d_t:+.2f}")
        # ==================================

        info["pitroad"] = pitroad
        info["pit_enter"] = pit_enter
        info["pit_exit"] = pit_exit
        info["pit_executed"] = pit_executed
        info["ell"] = float(ell_t)
        info["d_t"] = float(d_t)

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

            # ==== wear-dependent steering and throttle ====
            # g(wear) in [steering_grip_min, 1.0]
            grip = 1.0 - self.steering_wear_strength * self._wear
            grip = float(np.clip(grip, self.steering_grip_min, 1.0))
            steer *= grip

            # Throttle (acceleration) also affected by grip
            gas *= grip

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

            # Same grip logic for discrete mode
            grip = 1.0 - self.steering_wear_strength * self._wear
            grip = float(np.clip(grip, self.steering_grip_min, 1.0))
            steer *= grip

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
        """
        Estimate signed track curvature ahead of the car by looking at how the
        road orientation (beta) changes over the next few tiles.

        Positive kappa_t  -> turning one way
        Negative kappa_t  -> turning the other way
        """
        env = self._env.unwrapped
        track = getattr(env, "track", None)

        # Need at least 2 tiles to define a turn
        if not track or len(track) < 2:
            return 0.0

        n = len(track)
        lookahead_tiles = 12 # how many segments ahead to average over

        # Start from the tile closest to the car
        idx = self._nearest_tile_index()

        total_delta = 0.0
        samples = 0

        for k in range(lookahead_tiles):
            i0 = (idx + k) % n
            i1 = (idx + k + 1) % n

            _, beta0, _, _ = track[i0]
            _, beta1, _, _ = track[i1]

            d_beta = self._angle_diff(beta1, beta0) # in [-pi, pi]
            total_delta += d_beta
            samples += 1

        if samples == 0:
            return 0.0

        # Average turn per segment (radians)
        avg_delta = total_delta / float(samples)

        # Normalize to [-1, 1] by dividing by pi, then scale down to about [-0.05, 0.05]
        kappa = (avg_delta / math.pi) * 0.05

        return float(kappa)


    def _is_infield_from_offset(self):
        # Placeholder implementation
        # return False
        # use lateral offset d_t from the track centerline. If |d_t| > 4.0 meters, treat as infield/off-track.
        d_t = self._compute_offset()
        return abs(d_t) > 4.0
    
    def _nearest_tile_index(self):
        """
        Index of the track tile whose center is closest 
        to the car (aligns pit logic with painted tiles).
        """
        env = self._env.unwrapped
        track = env.track
        if not track:
            return 0
        car_x, car_y = float(env.car.hull.position[0]), float(env.car.hull.position[1])
        best_i, best_d2 = 0, float("inf")
        for i, (_, _, cx, cy) in enumerate(track):
            dx, dy = car_x - cx, car_y - cy
            d2 = dx*dx + dy*dy
            if d2 < best_d2:
                best_d2, best_i = d2, i
        return best_i
    
    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """
        Smallest signed difference between two angles a and b, in radians,
        wrapped into [-pi, pi].
        """
        d = a - b
        # Wrap using atan2(sin, cos) for numerical stability
        return math.atan2(math.sin(d), math.cos(d))
    
    def _sector_state_by_index(self, tile_idx: int):
        """
        Given a tile index, return (in_sector, sector_idx) 
        consistent with paint logic where prog = i / n.
        """
        env = self._env.unwrapped
        n = max(1, len(env.track))
        count = max(1, int(self._pit_count))
        step = 1.0 / float(count)                 # sector length in progress units
        width = float(self._pit_sector_width)     # painted width in progress units

        prog = tile_idx / float(n)
        sector_idx = (tile_idx * count) // n
        start = sector_idx * step

        # tiny epsilon eliminates fencepost disagreement
        eps = 1e-9
        in_sector = (start - eps <= prog <= start + width + eps)
        return in_sector, int(sector_idx)
    
    def _sync_pit_bounds(self):
        """
        Set lateral pit detection [d_min, d_max] to exactly 
        match how the pit strip is drawn (centered or sided).
        """
        if self._pit_center_test:
            half = float(abs(self._pit_width)) / 2.0
            self._pit_d_min, self._pit_d_max = -half, +half
        else:
            inner = float(abs(self._pit_inner_offset))
            width = float(abs(self._pit_width))
            d_hi = self._pit_sign * inner
            d_lo = self._pit_sign * (inner + width)
            self._pit_d_min, self._pit_d_max = (min(d_lo, d_hi), max(d_lo, d_hi))

    def _is_in_pit(self, d_t=None, ell_t=None):
        """
        Return True iff current lateral offset is within 
        pit bounds and the current tile lies in a pit sector.
        """
        try:
            if d_t is None:
                d_t = self._compute_offset()
            tile_idx = self._nearest_tile_index()
            in_sector, _ = self._sector_state_by_index(tile_idx)
            in_strip = (self._pit_d_min <= d_t <= self._pit_d_max)
            return bool(in_sector and in_strip)
        except Exception:
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

    def _build_pit_polys(self):
        """
        Build painted pit quads on the track.
        """
        env = self._env.unwrapped
        track = getattr(env, "track", None)
        if not track or len(track) < 2:
            return

        try:
            if self._pit_polys_count > 0:
                env.road_poly = env.road_poly[:-self._pit_polys_count]
                self._pit_polys_count = 0
        except Exception:
            self._pit_polys_count = 0

        n = len(track)
        count = max(1, int(self._pit_count))
        step = 1.0 / float(count)
        width_prog = float(self._pit_sector_width)

        PIT_COLOR = (255, 140, 0) # orange

        def lateral_offset(cx, cy, beta, offset_m):
            nx, ny = math.cos(beta), math.sin(beta)
            return (cx + offset_m * nx, cy + offset_m * ny)

        added = 0
        for i in range(n - 1):
            prog = i / float(n)
            k = int((prog % 1.0) / step)
            start = k * step
            if not (start <= prog <= start + width_prog):
                continue

            _, beta_i, cx_i, cy_i = track[i]
            _, beta_j, cx_j, cy_j = track[i + 1]

            if self._pit_center_test:
                half = float(abs(self._pit_width)) / 2.0
                # left/right edges around the centerline
                ix_i, iy_i = lateral_offset(cx_i, cy_i, beta_i, -half)
                ox_i, oy_i = lateral_offset(cx_i, cy_i, beta_i, +half)
                ix_j, iy_j = lateral_offset(cx_j, cy_j, beta_j, -half)
                ox_j, oy_j = lateral_offset(cx_j, cy_j, beta_j, +half)
            else:
                inner = float(abs(self._pit_inner_offset))
                strip_w = float(abs(self._pit_width))
                # right side using sign
                ix_i, iy_i = lateral_offset(cx_i, cy_i, beta_i, self._pit_sign * inner)
                ox_i, oy_i = lateral_offset(cx_i, cy_i, beta_i, self._pit_sign * (inner + strip_w))
                ix_j, iy_j = lateral_offset(cx_j, cy_j, beta_j, self._pit_sign * inner)
                ox_j, oy_j = lateral_offset(cx_j, cy_j, beta_j, self._pit_sign * (inner + strip_w))

            quad = [(ix_i, iy_i), (ox_i, oy_i), (ox_j, oy_j), (ix_j, iy_j)]
            try:
                env.road_poly.append((quad, PIT_COLOR))
                added += 1
            except Exception:
                break

        self._pit_polys_count = added

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

