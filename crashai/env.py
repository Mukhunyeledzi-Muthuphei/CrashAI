"""Gymnasium environment: the network takes over only when a crash is about to happen.

Episode flow:
  1. reset(): a scenario is generated and a simulated driver drives the car.
  2. The trigger fires when the perceived time-to-collision drops below `ttc_trigger`
     (or the car starts to slide). reset() returns the observation at that moment.
  3. step(): the network controls the car (steering, 4 brakes, throttle; depends on `controls`).
  4. The episode ends with a crash, a stop, a hand-back to the driver once no collision is
     predicted and the car is stable, or a timeout.

Control layouts (`controls`):
  "full"        6 actions: steer, brake FL, FR, RL, RR, throttle   (the idea)
  "pedal"       3 actions: steer, one brake pedal, throttle        (what a human has; ESC on)
  "brakes_only" 5 actions: brake FL, FR, RL, RR, throttle          (steering frozen)
  "driver"      no network: the driver keeps driving               (no intervention baseline)
All actions are in [-1, 1].
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from crashai import scenarios as sc
from crashai.geometry import box_corners, to_body, wrap_angle
from crashai.params import VehicleParams
from crashai.sensors import Perception, kind_index
from crashai.threat import obb_overlap, objects_to_array, time_to_collision, ttc_kernel
from crashai.vehicle import Vehicle
from crashai.world import LANE_WIDTH, OBJECT_TYPES

CONTROL_DT = 0.05  # 20 Hz decisions
COLLISION_DT = 0.01  # collisions checked every 10 ms
N_OBJ = 8  # object slots in the observation
OBJ_FEATS = 9 + len(OBJECT_TYPES)
EGO_FEATS = 30

ACTION_DIMS = {"full": 6, "pedal": 3, "brakes_only": 5, "driver": 1}


class CrashAvoidEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        controls: str = "full",
        scenario_kinds=sc.SCENARIOS,
        surface: str | None = None,
        ttc_trigger: float = 1.8,
        max_intervention: float = 8.0,
        use_abs: bool | None = None,
        use_esc: bool | None = None,
        record: bool = False,
        params: VehicleParams | None = None,
    ):
        assert controls in ACTION_DIMS, controls
        self.controls = controls
        self.scenario_kinds = tuple(scenario_kinds)
        self.surface = surface
        self.ttc_trigger = ttc_trigger
        self.max_intervention = max_intervention
        self.use_abs = True if use_abs is None else use_abs
        self.use_esc = (controls in ("pedal", "driver")) if use_esc is None else use_esc
        self.record = record
        self.car = Vehicle(params)
        self.car.use_abs = self.use_abs
        self.car.use_esc = self.use_esc

        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIMS[controls],), dtype=np.float32)
        n_obs = EGO_FEATS + N_OBJ * OBJ_FEATS
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(n_obs,), dtype=np.float32)
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        kind = options.get("scenario")
        for _attempt in range(30):
            scen_kind = kind or self.scenario_kinds[self._rng.integers(len(self.scenario_kinds))]
            self.scen = sc.generate(self._rng, scen_kind, self.surface)
            if self._run_until_trigger():
                break
        else:
            raise RuntimeError("could not generate a scenario that triggers")
        self.car.use_abs = self.use_abs
        self.car.use_esc = self.use_esc
        self.steer_frozen = self.car.steer  # used by "brakes_only"
        self.speed_at_takeover = self.car.speed
        return self._obs(), self._info()

    def _setup(self):
        s = self.scen
        self.road = s.road
        self.obstacles = s.obstacles
        self.driver = s.driver
        # While the human drives, the car has standard ABS + ESC. The chosen system takes over at the trigger.
        self.car.use_abs = True
        self.car.use_esc = True
        self.car.set_road(s.road.base_mu, s.road.patches)
        self.car.reset(x=0.0, y=s.ego_y, psi=0.0, speed=s.ego_speed)
        self.perception = Perception(self._rng, s.weather, s.latency_frames)
        self.t = 0.0
        self.t_takeover = None
        self.frames = []
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_controls = (0.0, np.zeros(4), 0.0)
        self.handback_timer = 0.0
        self.handed_back = False
        self.crashed_after_handback = False
        self.outcome = None
        self.impact_speed = 0.0
        self.driver_input = (0.0, 0.0, 0.0)
        self.perceived = self.perception.update(self.car, self.obstacles, CONTROL_DT)
        self.ttc_perceived = time_to_collision(self.car, self._perceived_for_ttc())[0]

    def _run_until_trigger(self) -> bool:
        """Let the driver drive until the system would step in. False if the scenario is unusable."""
        self._setup()
        for _ in range(int(8.0 / CONTROL_DT)):
            if self._triggered():
                self.t_takeover = self.t
                return True
            steer, pedal, throttle = self.driver.act(self.t, self.car, self.ttc_perceived)
            self.driver_input = (steer, pedal, throttle)
            self.car.set_controls(steer, pedal * self.car.p.max_brake, throttle)
            if self._advance():
                return False  # crashed before the trigger: unusable
        return False

    def _triggered(self):
        if self.ttc_perceived < self.ttc_trigger:
            return True
        return self.car.speed > 8.0 and abs(self.car.sideslip) > math.radians(8)

    # ------------------------------------------------------------------ step
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        p = self.car.p
        if self.controls == "full":
            steer = float(action[0]) * p.max_steer
            brakes = (action[1:5] + 1.0) / 2.0 * p.max_brake
            throttle = float(action[5] + 1.0) / 2.0
        elif self.controls == "pedal":
            steer = float(action[0]) * p.max_steer
            brakes = float(action[1] + 1.0) / 2.0 * p.max_brake
            throttle = float(action[2] + 1.0) / 2.0
        elif self.controls == "brakes_only":
            steer = self.steer_frozen
            brakes = (action[0:4] + 1.0) / 2.0 * p.max_brake
            throttle = float(action[4] + 1.0) / 2.0
        else:  # driver keeps driving
            steer, pedal, throttle = self.driver.act(self.t, self.car, self.ttc_perceived)
            brakes = pedal * p.max_brake
        self.car.set_controls(steer, brakes, throttle)
        self.prev_controls = (steer, np.asarray(brakes, dtype=float) * np.ones(4), throttle)

        phi_before = self._potential()
        crashed = self._advance()

        reward = -0.02
        reward -= 0.02 * float(np.sum((action - self.last_action) ** 2))
        reward -= 0.5 * max(0.0, abs(self.car.sideslip) - 0.05)
        terminated = False
        truncated = False

        if crashed:
            terminated = True
            if self.outcome == "crash":
                reward -= 50.0 + 3.0 * self.impact_speed
            else:  # guardrail
                reward -= 50.0 + 1.5 * self.impact_speed
        else:
            reward += 0.99 * self._potential() - phi_before
            stable = (abs(self.car.sideslip) < math.radians(3) and abs(self.car.yaw_rate) < 0.25
                      and abs(wrap_angle(self.car.psi)) < math.radians(12))
            if self.car.speed < 0.5:
                self.outcome = "stopped"
            elif math.isinf(self.ttc_perceived) and stable and self._clear_along_lane():
                self.handback_timer += CONTROL_DT
                if self.handback_timer >= 0.5:
                    self.outcome = "handback"
            else:
                self.handback_timer = 0.0
            if self.outcome == "handback" and self._after_handback():
                # the hand-back was premature: the driver crashed soon after
                terminated = True
                self.crashed_after_handback = True
                reward -= 50.0 + 3.0 * self.impact_speed
            elif self.outcome in ("stopped", "handback"):
                terminated = True
                dy = abs(self.car.y - self.road.lane_center(self.road.nearest_lane(self.car.y)))
                reward += 10.0 - 3.0 * min(1.0, dy / (LANE_WIDTH / 2))
            elif self.t - self.t_takeover >= self.max_intervention:
                self.outcome = "timeout"
                truncated = True

        self.last_action = action
        return self._obs(), 0.1 * reward, terminated, truncated, self._info()

    # ------------------------------------------------------------------ simulation
    def _clear_along_lane(self) -> bool:
        """No predicted collision if the car simply drives straight down the road from here.
        (The normal TTC check follows the car's current curve, which a small twitch can fool.)"""
        objs = self._perceived_for_ttc()
        if not objs:
            return True
        car = self.car
        ttc, _ = ttc_kernel(car.x, car.y, 0.0, car.speed, 0.0, car.p.length, car.p.width,
                            objects_to_array(objs), 4.0, 0.05, 0.2)
        return math.isinf(ttc)

    def _after_handback(self, duration=4.0) -> bool:
        """Check a hand-back: an alert driver (just warned by the system) drives on for a few
        seconds, keeping the nearest lane and braking quickly if a threat appears.
        Returns True if the car crashes. Stops early if the car comes to a stop."""
        from crashai.driver import Driver

        self.handed_back = True
        self.car.use_abs = True
        self.car.use_esc = True
        lane_y = self.road.lane_center(self.road.nearest_lane(self.car.y))
        driver = Driver(target_y=lane_y, target_speed=self.car.speed, distracted=False,
                        reaction_time=0.3, brake_strength=0.9)
        for _ in range(int(duration / CONTROL_DT)):
            steer, pedal, throttle = driver.act(self.t, self.car, self.ttc_perceived)
            self.car.set_controls(steer, pedal * self.car.p.max_brake, throttle)
            if self._advance():
                return True
            if self.car.speed < 0.5:
                break
        return False

    def _advance(self) -> bool:
        """Advance the world by one control step. Returns True if the car crashed."""
        n = int(round(CONTROL_DT / COLLISION_DT))
        for _ in range(n):
            self.car.step(COLLISION_DT)
            self.t += COLLISION_DT
            for o in self.obstacles:
                o.step(self.t, COLLISION_DT)
            if self._check_collision():
                self._record()
                return True
        self.perceived = self.perception.update(self.car, self.obstacles, CONTROL_DT)
        self.ttc_perceived = time_to_collision(self.car, self._perceived_for_ttc())[0]
        self._record()
        return False

    def _perceived_for_ttc(self):
        return [o for o in self.perceived if o.confidence > 0.3]

    def _check_collision(self) -> bool:
        car = self.car
        p = car.p
        for o in self.obstacles:
            if obb_overlap(car.x, car.y, car.psi, p.length, p.width, o.x, o.y, o.psi, o.length, o.width):
                rel = car.world_velocity - o.velocity()
                self.impact_speed = float(np.hypot(*rel))
                self.outcome = "crash"
                self.hit = o.kind
                return True
        corners = box_corners(car.x, car.y, car.psi, p.length, p.width)
        if corners[:, 1].min() < self.road.y_min or corners[:, 1].max() > self.road.y_max:
            self.impact_speed = float(car.speed)
            self.outcome = "guardrail"
            self.hit = "guardrail"
            return True
        return False

    def _potential(self):
        """Shaping: being close to a predicted collision is bad (uses the true world)."""
        ttc, _ = time_to_collision(self.car, self.obstacles_as_perceived_truth(), horizon=3.0, dt=0.1)
        return -2.0 * max(0.0, 1.0 - ttc / 3.0)

    def obstacles_as_perceived_truth(self):
        return [_TrueView(o) for o in self.obstacles]

    # ------------------------------------------------------------------ observation
    def _obs(self):
        car = self.car
        p = car.p
        ax, ay = car.accel
        mu_est = float(np.clip(np.mean(car.mu_under_wheels) + self._rng.normal(0, 0.08), 0.05, 1.2))
        lane = self.road.nearest_lane(car.y)
        steer_d, pedal_d, thr_d = self.driver_input
        t_int = 0.0 if self.t_takeover is None else self.t - self.t_takeover
        ego = [
            car.vx / 30.0, car.vy / 5.0, car.yaw_rate, ax / 10.0, ay / 10.0,
            car.steer / p.max_steer, car.sideslip / 0.3,
            *np.clip(car.wheel_slip, -1, 1.5),
            *(car.brake_torques / p.max_brake),
            self.prev_controls[2],
            mu_est,
            (car.y - self.road.lane_center(lane)) / (LANE_WIDTH / 2),
            wrap_angle(car.psi) / 0.5,
            (car.y - self.road.y_min) / 10.0,
            (self.road.y_max - car.y) / 10.0,
            float(lane > 0),
            float(lane < self.road.n_lanes - 1),
            steer_d / p.max_steer, pedal_d, thr_d,
            t_int / self.max_intervention,
            min(self.ttc_perceived, 4.0) / 4.0,
            float(car.esc_active),
            self.road.n_lanes / 3.0,
        ]
        assert len(ego) == EGO_FEATS, len(ego)

        vwx, vwy = car.world_velocity
        objs = []
        for o in self.perceived:
            bx, by = to_body(o.x - car.x, o.y - car.y, car.psi)
            objs.append((math.hypot(bx, by), o, bx, by))
        objs.sort(key=lambda z: z[0])
        feats = np.zeros((N_OBJ, OBJ_FEATS), dtype=np.float32)
        for k, (_, o, bx, by) in enumerate(objs[:N_OBJ]):
            rvx, rvy = to_body(o.vx - vwx, o.vy - vwy, car.psi)
            onehot = np.zeros(len(OBJECT_TYPES))
            onehot[kind_index(o.kind)] = 1.0
            feats[k, :9] = (1.0, bx / 50.0, by / 10.0, rvx / 30.0, rvy / 10.0, o.length / 10.0,
                            o.width / 3.0, wrap_angle(o.psi - car.psi) / math.pi, o.confidence)
            feats[k, 9:] = onehot
        obs = np.concatenate([np.asarray(ego, dtype=np.float32), feats.ravel()])
        return np.clip(obs, -10.0, 10.0)

    def _info(self):
        return {
            "scenario": self.scen.kind,
            "surface": self.scen.meta["surface"],
            "mu": self.scen.meta["mu"],
            "split_mu": self.scen.meta["split_mu"],
            "weather": self.scen.weather,
            "outcome": self.outcome,
            "crashed_after_handback": self.crashed_after_handback,
            "impact_speed": self.impact_speed,
            "speed_at_takeover": getattr(self, "speed_at_takeover", None),
            "t_intervention": 0.0 if self.t_takeover is None else self.t - self.t_takeover,
            "ttc": self.ttc_perceived,
        }

    # ------------------------------------------------------------------ recording
    def _record(self):
        if not self.record:
            return
        car = self.car
        self.frames.append({
            "t": self.t,
            "phase": "driver" if self.t_takeover is None or self.controls == "driver" or self.handed_back else "ai",
            "ego": (car.x, car.y, car.psi),
            "speed": car.speed,
            "steer": car.steer,
            "brakes": car.brake_torques / car.p.max_brake,
            "throttle": float(self.car.cmd[5]),
            "esc": car.esc_active,
            "ttc": self.ttc_perceived,
            "wheel_speed": car.wheel_speeds.copy(),
            "slip": car.wheel_slip,
            "alpha": car.diag[4:8].copy(),
            "fx": car.diag[12:16].copy(),
            "fy": car.diag[16:20].copy(),
            "fz": car.diag[8:12].copy(),
            "mu": car.mu_under_wheels,
            "yaw_rate": float(car.yaw_rate),
            "sideslip": float(car.sideslip),
            "obstacles": [(o.x, o.y, o.psi, o.length, o.width, o.kind) for o in self.obstacles],
            "perceived": [(o.x, o.y, o.psi, o.length, o.width, o.ghost) for o in self.perceived],
        })


class _TrueView:
    """Ground-truth obstacle in the same shape as a PerceivedObject (for the TTC kernel)."""

    __slots__ = ("x", "y", "vx", "vy", "psi", "length", "width")

    def __init__(self, o):
        self.x, self.y, self.psi, self.length, self.width = o.x, o.y, o.psi, o.length, o.width
        self.vx, self.vy = o.speed, o.vy
