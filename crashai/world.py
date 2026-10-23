"""The world: a straight multi-lane road with guardrails, friction patches and moving obstacles."""

import math
from dataclasses import dataclass, field

import numpy as np

from crashai.geometry import box_corners

LANE_WIDTH = 3.6
SHOULDER = 0.6

OBJECT_TYPES = ("car", "truck", "pedestrian", "cyclist", "debris")


@dataclass
class Road:
    """Straight road along +x. Lane i has its centre at y = i * LANE_WIDTH.

    Guardrails sit at y_min and y_max; touching one counts as a crash.
    """

    n_lanes: int = 2
    base_mu: float = 1.0
    surface: str = "dry"
    patches: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))  # x0,x1,y0,y1,mu

    @property
    def y_min(self):
        return -LANE_WIDTH / 2 - SHOULDER

    @property
    def y_max(self):
        return (self.n_lanes - 1) * LANE_WIDTH + LANE_WIDTH / 2 + SHOULDER

    def lane_center(self, i):
        return i * LANE_WIDTH

    def nearest_lane(self, y):
        return int(np.clip(round(y / LANE_WIDTH), 0, self.n_lanes - 1))

    def mu_at(self, x, y):
        mu = self.base_mu
        for x0, x1, y0, y1, m in self.patches:
            if x0 <= x <= x1 and y0 <= y <= y1:
                mu = m
        return mu


@dataclass
class Obstacle:
    """A road user or object. Moves along +x at `speed`, plus a lateral velocity `vy`.

    Scripted behaviour: braking at a set time, a smooth lane change, or crossing the road.
    """

    kind: str
    x: float
    y: float
    speed: float
    length: float
    width: float
    psi: float = 0.0
    vy: float = 0.0
    brake_at: float = math.inf
    brake_decel: float = 0.0
    lane_change_at: float = math.inf
    lane_change_dy: float = 0.0
    lane_change_time: float = 2.0
    cross_at: float = math.inf
    cross_speed: float = 0.0  # lateral speed while crossing (sign = direction)
    cross_stop_y: float | None = None

    def velocity(self):
        return np.array([self.speed, self.vy])

    def step(self, t, dt):
        if t >= self.brake_at:
            self.speed = max(0.0, self.speed - self.brake_decel * dt)

        vy = 0.0
        t_lc = t - self.lane_change_at
        if 0.0 <= t_lc <= self.lane_change_time:
            T = self.lane_change_time
            vy += self.lane_change_dy * math.pi / (2 * T) * math.sin(math.pi * t_lc / T)
        if t >= self.cross_at and self.cross_speed != 0.0:
            done = self.cross_stop_y is not None and (self.y - self.cross_stop_y) * math.copysign(
                1.0, self.cross_speed
            ) >= 0
            if not done:
                vy += self.cross_speed
        self.vy = vy

        self.x += self.speed * dt
        self.y += vy * dt
        if self.kind in ("car", "truck") and self.speed > 1.0:
            self.psi = math.atan2(vy, self.speed)

    def corners(self):
        return box_corners(self.x, self.y, self.psi, self.length, self.width)


def make_obstacle(kind, x, y, speed=0.0, rng=None, **kw):
    rng = rng or np.random.default_rng()
    sizes = {
        "car": (rng.uniform(4.2, 5.0), rng.uniform(1.75, 1.95)),
        "truck": (rng.uniform(8.0, 12.0), 2.5),
        "pedestrian": (0.5, 0.6),
        "cyclist": (1.8, 0.7),
        "debris": (rng.uniform(0.8, 2.0), rng.uniform(0.8, 1.6)),
    }
    length, width = sizes[kind]
    return Obstacle(kind=kind, x=x, y=y, speed=speed, length=length, width=width, **kw)
