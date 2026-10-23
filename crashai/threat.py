"""Threat assessment: time-to-collision (TTC) and collision checks.

TTC here is "path based": the ego car is predicted forward assuming constant speed and
yaw rate, every object at constant velocity, and TTC is the first time their rectangles
overlap. This handles crossing pedestrians and cut-ins, not just cars straight ahead.
"""

import math

import numpy as np
from numba import njit

# Object array columns for the kernels
OX, OY, OVX, OVY, OPSI, OL, OW = range(7)
OBJ_COLS = 7


@njit(cache=True)
def _corners(x, y, psi, length, width, out):
    c = math.cos(psi)
    s = math.sin(psi)
    hl = 0.5 * length
    hw = 0.5 * width
    lx = (hl, hl, -hl, -hl)
    ly = (hw, -hw, -hw, hw)
    for i in range(4):
        out[i, 0] = x + lx[i] * c - ly[i] * s
        out[i, 1] = y + lx[i] * s + ly[i] * c


@njit(cache=True)
def _separated(c1, c2):
    for which in range(2):
        cs = c1 if which == 0 else c2
        for e in range(2):
            ax = -(cs[e + 1, 1] - cs[e, 1])
            ay = cs[e + 1, 0] - cs[e, 0]
            min1 = 1e18
            max1 = -1e18
            min2 = 1e18
            max2 = -1e18
            for k in range(4):
                p1 = c1[k, 0] * ax + c1[k, 1] * ay
                p2 = c2[k, 0] * ax + c2[k, 1] * ay
                min1 = min(min1, p1)
                max1 = max(max1, p1)
                min2 = min(min2, p2)
                max2 = max(max2, p2)
            if max1 < min2 or max2 < min1:
                return True
    return False


@njit(cache=True)
def obb_overlap(x1, y1, p1, l1, w1, x2, y2, p2, l2, w2):
    if (x1 - x2) ** 2 + (y1 - y2) ** 2 > (0.5 * (l1 + w1 + l2 + w2)) ** 2:
        return False
    c1 = np.empty((4, 2))
    c2 = np.empty((4, 2))
    _corners(x1, y1, p1, l1, w1, c1)
    _corners(x2, y2, p2, l2, w2, c2)
    return not _separated(c1, c2)


@njit(cache=True)
def ttc_kernel(ex, ey, epsi, espeed, eyaw_rate, el, ew, objs, horizon, dt, margin):
    """Return (ttc, index of object). ttc = inf if no overlap within the horizon."""
    n = objs.shape[0]
    best = math.inf
    best_i = -1
    steps = int(horizon / dt) + 1
    for k in range(steps):
        t = k * dt
        # ego: constant speed, constant yaw rate (arc)
        if abs(eyaw_rate) > 1e-4:
            psi_t = epsi + eyaw_rate * t
            x_t = ex + espeed / eyaw_rate * (math.sin(psi_t) - math.sin(epsi))
            y_t = ey - espeed / eyaw_rate * (math.cos(psi_t) - math.cos(epsi))
        else:
            psi_t = epsi
            x_t = ex + espeed * math.cos(epsi) * t
            y_t = ey + espeed * math.sin(epsi) * t
        for i in range(n):
            ox = objs[i, OX] + objs[i, OVX] * t
            oy = objs[i, OY] + objs[i, OVY] * t
            if obb_overlap(x_t, y_t, psi_t, el + margin, ew + margin,
                           ox, oy, objs[i, OPSI], objs[i, OL] + margin, objs[i, OW] + margin):
                if t < best:
                    best = t
                    best_i = i
        if best_i >= 0:
            return best, best_i
    return best, best_i


def objects_to_array(objects) -> np.ndarray:
    """Objects must have x, y, vx, vy, psi, length, width (world frame)."""
    arr = np.zeros((len(objects), OBJ_COLS))
    for i, o in enumerate(objects):
        arr[i] = (o.x, o.y, o.vx, o.vy, o.psi, o.length, o.width)
    return arr


def time_to_collision(vehicle, objects, horizon=4.0, dt=0.05, margin=0.2):
    if len(objects) == 0:
        return math.inf, -1
    vx, vy = vehicle.world_velocity
    speed = math.hypot(vx, vy)
    heading = math.atan2(vy, vx) if speed > 0.5 else vehicle.psi
    return ttc_kernel(
        vehicle.x, vehicle.y, heading, speed, vehicle.yaw_rate,
        vehicle.p.length, vehicle.p.width, objects_to_array(objects), horizon, dt, margin,
    )
