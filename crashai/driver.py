"""A simple human driver model: keeps its lane and speed, reacts late (or not at all) to danger."""

import math
from dataclasses import dataclass

from crashai.geometry import to_body


@dataclass
class Driver:
    target_y: float
    target_speed: float
    distracted: bool = True  # a distracted driver never reacts before the system steps in
    reaction_time: float = 1.5  # s from the hazard appearing to starting to brake
    brake_strength: float = 0.6  # pedal position the driver eventually uses (0..1)
    hazard_seen_at: float | None = None

    def act(self, t, ego, ttc):
        # lane keeping: pure pursuit to a point ahead on the lane centre
        look = max(10.0, 1.0 * ego.speed)
        bx, by = to_body(look, self.target_y - ego.y, ego.psi)
        L = ego.p.wheelbase
        steer = math.atan2(2.0 * L * by, bx * bx + by * by)
        steer = max(-0.15, min(0.15, steer))

        # speed keeping
        err = self.target_speed - ego.speed
        throttle = max(0.0, min(1.0, 0.15 + 0.3 * err))
        pedal = 0.0
        if err < -2.0:
            throttle = 0.0
            pedal = min(0.3, -0.05 * err)

        # late reaction to a threat
        if not self.distracted and ttc < 3.0:
            if self.hazard_seen_at is None:
                self.hazard_seen_at = t
            if t - self.hazard_seen_at >= self.reaction_time:
                ramp = min(1.0, (t - self.hazard_seen_at - self.reaction_time) / 0.4)
                pedal = self.brake_strength * ramp
                throttle = 0.0
        return steer, pedal, throttle
