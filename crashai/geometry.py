"""Rectangle geometry: corners and overlap tests (separating axis theorem)."""

import math

import numpy as np


def box_corners(x, y, psi, length, width):
    """4x2 array of corner positions of a rectangle centred at (x, y) with heading psi."""
    c, s = math.cos(psi), math.sin(psi)
    hl, hw = length / 2, width / 2
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def boxes_overlap(c1: np.ndarray, c2: np.ndarray) -> bool:
    """True if two convex quadrilaterals (4x2 corner arrays) overlap."""
    for corners in (c1, c2):
        for i in range(2):  # rectangles: 2 unique edge normals each
            edge = corners[i + 1] - corners[i]
            axis = np.array([-edge[1], edge[0]])
            p1 = c1 @ axis
            p2 = c2 @ axis
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
    return True


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def to_body(dx, dy, psi):
    """Rotate a world-frame vector into a body frame with heading psi."""
    c, s = math.cos(psi), math.sin(psi)
    return dx * c + dy * s, -dx * s + dy * c
