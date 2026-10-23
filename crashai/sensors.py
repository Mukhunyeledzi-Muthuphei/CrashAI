"""Simulated sensor fusion output.

We don't simulate camera pixels or lidar points. A modern car's fusion software turns all its
sensors into an object list, so we build that list from the true world and then corrupt it:
correlated position/velocity noise, delay, missed detections (with track coasting), false
"ghost" objects, limited range (shorter in bad weather) and occlusion behind other objects.

Target sensor set (high-end 2025-26 car): 8 cameras (360 deg), 1 front lidar (~250 m),
1 front long-range radar (~200 m), 4 corner radars (~90 m), 12 ultrasonic (~5 m).
"""

import math
from dataclasses import dataclass

import numpy as np

from crashai.geometry import to_body
from crashai.world import OBJECT_TYPES

WEATHER = {
    #          range factor, noise factor, miss prob, ghost prob per frame
    "clear": (1.00, 1.0, 0.02, 0.005),
    "rain": (0.65, 1.6, 0.06, 0.015),
    "snow": (0.50, 2.0, 0.10, 0.020),
    "fog": (0.40, 1.5, 0.08, 0.010),
}


@dataclass
class PerceivedObject:
    """One entry of the fused object list (world frame, as the car believes it)."""

    x: float
    y: float
    vx: float
    vy: float
    psi: float
    length: float
    width: float
    kind: str
    confidence: float
    ghost: bool = False
    track_id: int = -1


class Perception:
    FRONT_RANGE = 220.0  # best of lidar / long-range radar / front camera
    SIDE_RANGE = 90.0  # corner radars + surround cameras
    PED_RANGE = 90.0  # small objects are harder to see
    FRONT_HALF_ANGLE = math.radians(60)

    def __init__(self, rng: np.random.Generator, weather: str = "clear", latency_frames: int = 1):
        self.rng = rng
        self.weather = weather
        self.range_f, self.noise_f, self.p_miss, self.p_ghost = WEATHER[weather]
        self.latency = latency_frames
        self.tracks: dict[int, dict] = {}
        self.ghosts: list[dict] = []
        self.buffer: list[list[PerceivedObject]] = []

    # --------------------------------------------------------------
    def _visible(self, ego, obstacles):
        """Indices of obstacles that are in range and not hidden behind others."""
        info = []
        for i, o in enumerate(obstacles):
            bx, by = to_body(o.x - ego.x, o.y - ego.y, ego.psi)
            dist = math.hypot(bx, by)
            bearing = math.atan2(by, bx)
            small = o.kind in ("pedestrian", "cyclist", "debris")
            if abs(bearing) < self.FRONT_HALF_ANGLE:
                rng_max = (self.PED_RANGE if small else self.FRONT_RANGE) * self.range_f
            else:
                rng_max = min(self.SIDE_RANGE, self.PED_RANGE if small else self.SIDE_RANGE) * self.range_f
            half = math.atan2(0.5 * max(o.width, min(o.length, 2.0)), max(dist, 0.5))
            info.append((i, dist, bearing, half, dist <= rng_max))
        visible = []
        for i, dist, bearing, half, in_range in info:
            if not in_range:
                continue
            # occlusion: sample bearings across the object, hidden if all are blocked by nearer objects
            samples = np.linspace(bearing - 0.8 * half, bearing + 0.8 * half, 5)
            blocked = 0
            for s in samples:
                for j, dj, bj, hj, _ in info:
                    if j != i and dj + 1.0 < dist and abs(math.remainder(s - bj, 2 * math.pi)) < hj:
                        blocked += 1
                        break
            if blocked < len(samples):
                visible.append(i)
        return visible

    def update(self, ego, obstacles, dt):
        rng = self.rng
        seen = set(self._visible(ego, obstacles))
        out = []
        sig_p = 0.12 * self.noise_f
        sig_v = 0.25 * self.noise_f
        for i, o in enumerate(obstacles):
            tr = self.tracks.get(i)
            detected = i in seen and rng.random() > self.p_miss
            if detected:
                dist = math.hypot(o.x - ego.x, o.y - ego.y)
                sp = sig_p * (1.0 + dist / 60.0)
                if tr is None:
                    tr = {"nx": 0.0, "ny": 0.0, "nvx": 0.0, "nvy": 0.0, "miss": 0, "age": 0,
                          "dl": rng.normal(0, 0.15), "dw": rng.normal(0, 0.08)}
                    self.tracks[i] = tr
                # correlated (AR1) noise, like a real tracker's slowly drifting error
                a = 0.8
                q = math.sqrt(1 - a * a)
                tr["nx"] = a * tr["nx"] + q * rng.normal(0, sp)
                tr["ny"] = a * tr["ny"] + q * rng.normal(0, sp * 0.6)
                tr["nvx"] = a * tr["nvx"] + q * rng.normal(0, sig_v)
                tr["nvy"] = a * tr["nvy"] + q * rng.normal(0, sig_v * 0.6)
                tr["miss"] = 0
                tr["age"] += 1
                v = o.velocity()
                tr["est"] = PerceivedObject(
                    x=o.x + tr["nx"], y=o.y + tr["ny"], vx=v[0] + tr["nvx"], vy=v[1] + tr["nvy"],
                    psi=o.psi, length=max(0.3, o.length + tr["dl"]), width=max(0.3, o.width + tr["dw"]),
                    kind=o.kind, confidence=min(1.0, 0.5 + 0.1 * tr["age"]), track_id=i,
                )
                out.append(tr["est"])
            elif tr is not None and "est" in tr:
                # coast the track for a few frames, then drop it
                tr["miss"] += 1
                if tr["miss"] > 4:
                    del self.tracks[i]
                    continue
                e = tr["est"]
                e = PerceivedObject(e.x + e.vx * dt, e.y + e.vy * dt, e.vx, e.vy, e.psi, e.length,
                                    e.width, e.kind, e.confidence * 0.7, track_id=i)
                tr["est"] = e
                out.append(e)

        # ghosts (radar reflections, misclassified shadows...)
        if rng.random() < self.p_ghost:
            bx = rng.uniform(15, 80)
            by = rng.uniform(-3, 3)
            c, s = math.cos(ego.psi), math.sin(ego.psi)
            self.ghosts.append({"life": int(rng.integers(2, 7)), "obj": PerceivedObject(
                x=ego.x + bx * c - by * s, y=ego.y + bx * s + by * c, vx=0.0, vy=0.0, psi=0.0,
                length=rng.uniform(0.5, 2.0), width=rng.uniform(0.5, 1.5), kind="debris",
                confidence=rng.uniform(0.15, 0.45), ghost=True)})
        for g in self.ghosts:
            g["life"] -= 1
            out.append(g["obj"])
        self.ghosts = [g for g in self.ghosts if g["life"] > 0]

        self.buffer.append(out)
        if len(self.buffer) > self.latency + 1:
            self.buffer.pop(0)
        return self.buffer[0]

    @property
    def current(self):
        return self.buffer[0] if self.buffer else []


def kind_index(kind: str) -> int:
    return OBJECT_TYPES.index(kind)
