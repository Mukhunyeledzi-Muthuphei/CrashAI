"""Randomised emergency scenarios.

Each scenario places the ego car on a road with a hazard set up so that a crash is a few
seconds away, plus random surrounding traffic that may block an escape lane. Road surface
(dry / wet / snow / ice), split-friction patches, weather and sensor latency are random too.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from crashai.driver import Driver
from crashai.world import LANE_WIDTH, Obstacle, Road, make_obstacle

SCENARIOS = ("stopped", "lead_brake", "cut_in", "pedestrian")

SURFACES = {
    #        prob,  mu range,        weather options
    "dry": (0.50, (0.90, 1.05), ("clear", "clear", "clear", "fog")),
    "wet": (0.25, (0.60, 0.80), ("rain",)),
    "snow": (0.15, (0.30, 0.45), ("snow", "clear")),
    "ice": (0.10, (0.12, 0.20), ("clear", "fog")),
}


@dataclass
class Scenario:
    kind: str
    road: Road
    ego_y: float
    ego_speed: float
    obstacles: list[Obstacle]
    driver: Driver
    weather: str
    latency_frames: int
    meta: dict = field(default_factory=dict)


def _surface(rng, surface=None):
    names = list(SURFACES)
    if surface is None:
        probs = np.array([SURFACES[n][0] for n in names])
        surface = names[rng.choice(len(names), p=probs / probs.sum())]
    _, (lo, hi), weathers = SURFACES[surface]
    mu = rng.uniform(lo, hi)
    weather = weathers[rng.integers(len(weathers))]
    return surface, mu, weather


def _split_patch(rng, road, ego_lane):
    """A low-grip patch covering one side of the ego lane (split-mu braking)."""
    x0 = rng.uniform(15, 50)
    x1 = x0 + rng.uniform(40, 150)
    yc = road.lane_center(ego_lane)
    if rng.random() < 0.5:
        y0, y1 = road.y_min - 1, yc  # right half and everything to the right
    else:
        y0, y1 = yc, road.y_max + 1
    mu = road.base_mu * rng.uniform(0.15, 0.5)
    return np.array([[x0, x1, y0, y1, mu]])


def _traffic(rng, road, ego_lane, v, exclude_lanes=(), p=0.45):
    """Random cars in lanes next to the ego car; they can block an escape."""
    obs = []
    for lane in range(road.n_lanes):
        if lane == ego_lane or lane in exclude_lanes:
            continue
        if abs(lane - ego_lane) == 1 and rng.random() < p:
            obs.append(make_obstacle("car", rng.uniform(-12, 35), road.lane_center(lane) + rng.normal(0, 0.15),
                                     speed=max(0.0, v + rng.uniform(-4, 3)), rng=rng))
    return obs


def generate(rng: np.random.Generator, kind: str | None = None, surface: str | None = None) -> Scenario:
    kind = kind or SCENARIOS[rng.integers(len(SCENARIOS))]
    surf, mu, weather = _surface(rng, surface)
    latency = int(rng.integers(1, 3))  # 50-100 ms

    if kind == "pedestrian":
        n_lanes = 3  # lane 0 holds parked cars
        ego_lane = 1
    else:
        n_lanes = int(rng.integers(2, 4))
        ego_lane = int(rng.integers(n_lanes))
    road = Road(n_lanes=n_lanes, base_mu=mu, surface=surf)
    if rng.random() < 0.2 and kind != "pedestrian":
        road.patches = _split_patch(rng, road, ego_lane)

    slow = {"dry": 1.0, "wet": 0.9, "snow": 0.7, "ice": 0.55}[surf]
    ego_y = road.lane_center(ego_lane) + rng.normal(0, 0.2)
    obstacles: list[Obstacle] = []
    driver_kw = {}

    if kind == "stopped":
        v = rng.uniform(14, 33) * slow
        what = rng.choice(["car", "car", "car", "truck", "debris"])
        o = make_obstacle(what, 0, road.lane_center(ego_lane) + rng.uniform(-0.6, 0.6), speed=0.0, rng=rng)
        o.x = v * rng.uniform(2.2, 3.4) + o.length / 2 + 2.4
        obstacles.append(o)
        obstacles += _traffic(rng, road, ego_lane, v)
        driver_kw = dict(distracted=True)

    elif kind == "lead_brake":
        v = rng.uniform(14, 33) * slow
        headway = rng.uniform(0.5, 1.4)
        o = make_obstacle(rng.choice(["car", "car", "truck"]), 0, road.lane_center(ego_lane) + rng.normal(0, 0.2),
                          speed=v + rng.uniform(-1, 1), rng=rng)
        o.x = v * headway + o.length / 2 + 2.4
        o.brake_at = rng.uniform(0.2, 1.0)
        o.brake_decel = min(rng.uniform(6.0, 9.5), 0.95 * mu * 9.81)
        obstacles.append(o)
        obstacles += _traffic(rng, road, ego_lane, v)
        driver_kw = dict(distracted=rng.random() < 0.5, reaction_time=rng.uniform(1.0, 2.0),
                         brake_strength=rng.uniform(0.4, 0.8))

    elif kind == "cut_in":
        if n_lanes < 2:
            road.n_lanes = n_lanes = 2
        v = rng.uniform(18, 33) * slow
        side = [lane for lane in (ego_lane - 1, ego_lane + 1) if 0 <= lane < n_lanes]
        from_lane = side[rng.integers(len(side))]
        o = make_obstacle(rng.choice(["car", "car", "car", "truck"]), rng.uniform(14, 35),
                          road.lane_center(from_lane), speed=max(3.0, v - rng.uniform(5, 14)), rng=rng)
        o.lane_change_at = rng.uniform(0.0, 0.6)
        o.lane_change_time = rng.uniform(1.2, 2.5)
        o.lane_change_dy = road.lane_center(ego_lane) - road.lane_center(from_lane) + rng.normal(0, 0.2)
        obstacles.append(o)
        obstacles += _traffic(rng, road, ego_lane, v, exclude_lanes=(from_lane,))
        driver_kw = dict(distracted=rng.random() < 0.5, reaction_time=rng.uniform(0.8, 1.6),
                         brake_strength=rng.uniform(0.4, 0.8))

    elif kind == "pedestrian":
        v = rng.uniform(8, 17) * slow
        # parked cars along lane 0, with a gap the pedestrian steps out of
        x_ped = v * rng.uniform(1.7, 2.8) + 2.4
        x = rng.uniform(-10, 5)
        while x < x_ped + 40:
            if abs(x - x_ped) > 3.2:
                obstacles.append(make_obstacle("car", x, road.lane_center(0) - 0.3 + rng.normal(0, 0.1), rng=rng))
            x += rng.uniform(5.8, 8.0)
        kind_p = "pedestrian" if rng.random() < 0.8 else "cyclist"
        ped = make_obstacle(kind_p, x_ped, road.lane_center(0) + rng.uniform(-0.2, 0.6), rng=rng)
        speed_lat = rng.uniform(1.2, 3.0) if kind_p == "pedestrian" else rng.uniform(3.0, 5.0)
        ped.cross_at = 0.0
        ped.cross_speed = speed_lat
        ped.cross_stop_y = road.lane_center(2) + rng.uniform(-1.0, 2.5)
        if kind_p == "cyclist":
            ped.psi = math.pi / 2
        obstacles.append(ped)
        if rng.random() < 0.4:  # traffic in the far lane
            obstacles.append(make_obstacle("car", rng.uniform(-5, 50), road.lane_center(2),
                                           speed=v + rng.uniform(-3, 3), rng=rng))
        driver_kw = dict(distracted=rng.random() < 0.6, reaction_time=rng.uniform(0.8, 1.5),
                         brake_strength=rng.uniform(0.5, 0.9))
    else:
        raise ValueError(f"unknown scenario {kind}")

    driver = Driver(target_y=road.lane_center(ego_lane), target_speed=v, **driver_kw)
    return Scenario(kind=kind, road=road, ego_y=ego_y, ego_speed=v, obstacles=obstacles,
                    driver=driver, weather=weather, latency_frames=latency,
                    meta={"surface": surf, "mu": mu, "ego_lane": ego_lane, "split_mu": len(road.patches) > 0})
