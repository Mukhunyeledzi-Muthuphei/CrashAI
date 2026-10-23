"""Rule-based baselines. They use the same perceived object list the network sees.

All of them run on a "full" control environment and output a 6-value action:
steer, brake FL, FR, RL, RR, throttle (each in [-1, 1]).
"""

import math

import numpy as np

from crashai.geometry import to_body
from crashai.threat import ttc_kernel, objects_to_array


class Controller:
    name = "controller"

    def reset(self, env):
        pass

    def act(self, env, obs):
        raise NotImplementedError


class BrakeStraight(Controller):
    """Full braking, steering wheel held straight. What most drivers do."""

    name = "brake_straight"

    def act(self, env, obs):
        return np.array([0.0, 1, 1, 1, 1, -1], dtype=np.float32)


class BrakeSwerve(Controller):
    """Full braking plus a scripted swerve into the lane with the most room (if any)."""

    name = "brake_swerve"

    def reset(self, env):
        self.target_y = None

    def _lane_ttc(self, env, y_c):
        car = env.car
        objs = [o for o in env.perceived if o.confidence > 0.3]
        if not objs:
            return math.inf
        vx, vy = car.world_velocity
        speed = math.hypot(vx, vy)
        ttc, _ = ttc_kernel(car.x, y_c, 0.0, speed, 0.0, car.p.length, car.p.width,
                            objects_to_array(objs), 4.0, 0.05, 0.4)
        return ttc

    def act(self, env, obs):
        car = env.car
        road = env.road
        if self.target_y is None:
            lane = road.nearest_lane(car.y)
            options = [lane] + [l for l in (lane - 1, lane + 1) if 0 <= l < road.n_lanes]
            scores = {l: self._lane_ttc(env, road.lane_center(l)) for l in options}
            best = max(options, key=lambda l: (scores[l], l == lane))
            # only leave the lane if it is clearly better
            if best != lane and scores[best] > scores[lane] + 0.3:
                self.target_y = road.lane_center(best)
            else:
                self.target_y = car.y
        look = max(8.0, 0.7 * car.speed)
        bx, by = to_body(look, self.target_y - car.y, car.psi)
        steer = math.atan2(2.0 * car.p.wheelbase * by, bx * bx + by * by)
        a_steer = float(np.clip(steer / car.p.max_steer, -1, 1))
        return np.array([a_steer, 1, 1, 1, 1, -1], dtype=np.float32)


class SB3Policy(Controller):
    """A trained Stable-Baselines3 policy."""

    def __init__(self, model, name):
        self.model = model
        self.name = name

    def act(self, env, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class DriverOnly(Controller):
    """No intervention: the (late or distracted) driver keeps driving. Needs controls='driver'."""

    name = "driver_only"

    def act(self, env, obs):
        return np.zeros(1, dtype=np.float32)
