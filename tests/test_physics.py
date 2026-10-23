import math

import numpy as np
import pytest

from crashai.vehicle import Vehicle


def brake_to_stop(v, max_t=15.0):
    t = 0.0
    while v.speed > 0.1 and t < max_t:
        v.step(0.01)
        t += 0.01
    return t


@pytest.mark.parametrize("mu", [1.0, 0.7, 0.3])
def test_abs_braking_decel_matches_grip(mu):
    v = Vehicle()
    v.set_road(mu)
    v.reset(speed=25.0)
    v.set_controls(0.0, v.p.max_brake, 0.0)
    t = brake_to_stop(v)
    decel = 25.0 / t
    # ABS should get close to the friction limit mu * g, never above it (plus a little drag)
    assert 0.80 * mu * 9.81 < decel < 1.05 * mu * 9.81 + 0.3


def test_abs_stops_shorter_than_locked_wheels():
    dist = {}
    for abs_on in (True, False):
        v = Vehicle()
        v.use_abs = abs_on
        v.reset(speed=30.0)
        v.set_controls(0.0, v.p.max_brake, 0.0)
        brake_to_stop(v)
        dist[abs_on] = v.x
    assert dist[True] < dist[False] - 2.0


def test_locked_wheels_lose_steering():
    y = {}
    for abs_on in (True, False):
        v = Vehicle()
        v.use_abs = abs_on
        v.reset(speed=25.0)
        v.set_controls(math.radians(8), v.p.max_brake, 0.0)
        for _ in range(100):
            v.step(0.01)
        y[abs_on] = v.y
    assert y[True] > 2 * y[False]


@pytest.mark.parametrize("side", ["left", "right"])
def test_differential_braking_turns_the_car(side):
    """The core idea: braking one side yaws the car toward that side."""
    v = Vehicle()
    v.reset(speed=25.0)
    tb = v.p.max_brake.copy()
    if side == "left":
        tb[[1, 3]] = 0.0
    else:
        tb[[0, 2]] = 0.0
    v.set_controls(0.0, tb, 0.0)
    for _ in range(50):
        v.step(0.01)
    sign = 1 if side == "left" else -1
    assert sign * v.psi > math.radians(3)
    assert sign * v.y > 0.1


def test_cornering_limited_by_friction():
    for mu in (1.0, 0.4):
        v = Vehicle()
        v.set_road(mu)
        v.reset(speed=25.0)
        v.set_controls(math.radians(35), np.zeros(4), 0.0)
        max_ay = 0.0
        for _ in range(300):
            v.step(0.01)
            max_ay = max(max_ay, abs(v.accel[1]))
        assert max_ay < 1.05 * mu * 9.81


def test_open_differential_brake_trick():
    """On split grip, braking the spinning wheel lets the other wheel drive the car."""
    speeds = {}
    for brake_spinning in (False, True):
        v = Vehicle()
        # ice under the left wheels (y > 0), dry under the right
        v.set_road(1.0, np.array([[-100, 1000, 0.0, 5.0, 0.1]]))
        v.reset(speed=2.0)
        tb = np.zeros(4)
        if brake_spinning:
            tb[0] = 400.0  # front-left is on ice and spins
        v.set_controls(0.0, tb, 1.0)
        for _ in range(200):
            v.step(0.01)
        speeds[brake_spinning] = v.speed
    assert speeds[True] > speeds[False] + 0.5


def test_no_nans_in_violent_maneuver():
    v = Vehicle()
    v.set_road(0.3)
    v.reset(speed=33.0)
    rng = np.random.default_rng(0)
    for _ in range(400):
        v.set_controls(rng.uniform(-0.6, 0.6), rng.uniform(0, 1, 4) * v.p.max_brake, rng.uniform())
        v.step(0.05)
        assert np.all(np.isfinite(v.S))
