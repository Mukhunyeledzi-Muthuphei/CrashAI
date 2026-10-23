"""Vehicle parameters for a typical mid-size front-wheel-drive car."""

from dataclasses import dataclass

import numpy as np

# Wheel order used everywhere: front-left, front-right, rear-left, rear-right.
FL, FR, RL, RR = 0, 1, 2, 3
WHEEL_NAMES = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class VehicleParams:
    # Body
    mass: float = 1500.0  # kg
    yaw_inertia: float = 2500.0  # kg m^2
    wheelbase: float = 2.70  # m
    front_weight: float = 0.60  # share of static weight on the front axle
    track: float = 1.55  # m
    cg_height: float = 0.55  # m
    length: float = 4.80  # m (body, for collisions)
    width: float = 1.85  # m

    # Wheels and tires
    wheel_radius: float = 0.31  # m
    wheel_inertia: float = 1.2  # kg m^2 per wheel
    # Pacejka "Magic Formula" shape (dry asphalt-like); peak scaled by road friction mu.
    tire_B: float = 10.0
    tire_C: float = 1.9
    tire_E: float = 0.90

    # Drivetrain: FWD, open differential, single throttle.
    max_wheel_torque: float = 4000.0  # N m total at the front axle
    max_power: float = 140e3  # W
    engine_inertia: float = 1.0  # kg m^2 reflected to the axle

    # Brakes (per wheel)
    max_brake_front: float = 2500.0  # N m
    max_brake_rear: float = 1200.0  # N m
    brake_build_time: float = 0.15  # s from zero to full torque
    brake_release_time: float = 0.08  # s from full to zero

    # Steering
    max_steer: float = np.deg2rad(35.0)  # rad, road-wheel angle
    max_steer_rate: float = np.deg2rad(60.0)  # rad/s at the road wheels

    # Resistances
    drag_coeff_area: float = 0.65  # Cd * A, m^2
    air_density: float = 1.2
    rolling_resistance: float = 0.012

    # ABS (low-level slip limiter, runs every physics substep)
    abs_slip: float = 0.12  # brake pressure is dumped when slip goes below -abs_slip
    abs_dump_time: float = 0.04  # s to dump full torque

    # ESC (simple yaw-rate controller using individual brakes)
    esc_gain: float = 6000.0  # N m of brake torque per rad/s of yaw error
    esc_deadband: float = 0.06  # rad/s
    understeer_gradient: float = 0.0025  # rad / (m/s^2), for the reference yaw rate

    @property
    def a(self) -> float:
        """Distance from centre of gravity to front axle."""
        return self.wheelbase * (1.0 - self.front_weight)

    @property
    def b(self) -> float:
        """Distance from centre of gravity to rear axle."""
        return self.wheelbase * self.front_weight

    @property
    def max_brake(self) -> np.ndarray:
        return np.array(
            [self.max_brake_front, self.max_brake_front, self.max_brake_rear, self.max_brake_rear]
        )

    def as_array(self) -> np.ndarray:
        """Pack parameters into a flat array for the compiled physics kernel (see physics.py)."""
        return np.array(
            [
                self.mass,
                self.yaw_inertia,
                self.a,
                self.b,
                self.track,
                self.cg_height,
                self.wheel_radius,
                self.wheel_inertia,
                self.tire_B,
                self.tire_C,
                self.tire_E,
                self.max_wheel_torque,
                self.max_power,
                self.engine_inertia,
                self.max_brake_front,
                self.max_brake_rear,
                self.brake_build_time,
                self.brake_release_time,
                self.max_steer,
                self.max_steer_rate,
                0.5 * self.air_density * self.drag_coeff_area,
                self.rolling_resistance,
                self.abs_slip,
                self.abs_dump_time,
                self.esc_gain,
                self.esc_deadband,
                self.understeer_gradient,
            ],
            dtype=np.float64,
        )
