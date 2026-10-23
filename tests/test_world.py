import math

import numpy as np

from crashai.geometry import box_corners, boxes_overlap
from crashai.threat import obb_overlap, ttc_kernel
from crashai.vehicle import Vehicle
from crashai.threat import time_to_collision
from crashai.world import make_obstacle


def test_overlap_matches_reference():
    rng = np.random.default_rng(0)
    for _ in range(500):
        a = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.5, 5), rng.uniform(0.5, 2))
        b = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.5, 5), rng.uniform(0.5, 2))
        ref = boxes_overlap(box_corners(*a), box_corners(*b))
        assert obb_overlap(*a, *b) == ref


def test_ttc_straight_ahead():
    # ego at 20 m/s, stationary 4 m object whose rear is 40 m ahead of ego's front
    objs = np.array([[0 + 2.4 + 40 + 2.0, 0, 0, 0, 0, 4.0, 1.8]])
    ttc, i = ttc_kernel(0, 0, 0, 20.0, 0.0, 4.8, 1.85, objs, 4.0, 0.01, 0.0)
    assert i == 0
    assert abs(ttc - 2.0) < 0.02


def test_ttc_adjacent_lane_is_safe():
    objs = np.array([[30, 3.6, 0, 0, 0, 4.5, 1.8]])
    ttc, i = ttc_kernel(0, 0, 0, 20.0, 0.0, 4.8, 1.85, objs, 4.0, 0.05, 0.2)
    assert math.isinf(ttc) and i == -1


def test_crossing_pedestrian_ttc():
    v = Vehicle()
    v.reset(speed=10.0)
    ped = make_obstacle("pedestrian", 22.4, -3.0, rng=np.random.default_rng(0))

    class P:  # perceived-object shaped
        x, y, vx, vy, psi, length, width = ped.x, ped.y, 0.0, 1.5, 0.0, ped.length, ped.width

    ttc, _ = time_to_collision(v, [P()], margin=0.0)
    assert 1.0 < ttc < 2.5
