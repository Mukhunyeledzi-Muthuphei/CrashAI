"""Python-side wrapper around the compiled physics kernel."""

import math

import numpy as np

from crashai import physics as ph
from crashai.params import VehicleParams

PHYSICS_DT = 0.002  # s


class Vehicle:
    def __init__(self, params: VehicleParams | None = None):
        self.p = params or VehicleParams()
        self._P = self.p.as_array()
        self.S = np.zeros(ph.STATE_SIZE)
        self.A = np.zeros(ph.ACT_SIZE)
        self.cmd = np.zeros(ph.CMD_SIZE)
        self.diag = np.zeros(ph.DIAG_SIZE)
        self.base_mu = 1.0
        self.patches = np.zeros((0, 5))
        self.use_abs = True
        self.use_esc = False

    # ----- setup
    def reset(self, x=0.0, y=0.0, psi=0.0, speed=20.0, delta=0.0):
        self.S[:] = 0.0
        self.S[ph.SX] = x
        self.S[ph.SY] = y
        self.S[ph.SPSI] = psi
        self.S[ph.SVX] = speed
        self.S[ph.SW0 : ph.SW0 + 4] = speed / self.p.wheel_radius
        self.A[:] = 0.0
        self.A[ph.A_DELTA] = delta
        self.cmd[:] = 0.0
        self.cmd[ph.C_DELTA] = delta
        self.diag[:] = 0.0
        self.diag[ph.D_FZ : ph.D_FZ + 4] = self.static_loads()
        self.diag[ph.D_MU : ph.D_MU + 4] = self.base_mu

    def set_road(self, base_mu: float, patches: np.ndarray | None = None):
        self.base_mu = float(base_mu)
        self.patches = np.zeros((0, 5)) if patches is None else np.asarray(patches, dtype=np.float64)

    def static_loads(self):
        p = self.p
        f = p.mass * ph.G * p.b / p.wheelbase / 2
        r = p.mass * ph.G * p.a / p.wheelbase / 2
        return np.array([f, f, r, r])

    # ----- control
    def set_controls(self, steer, brakes, throttle):
        """steer: road-wheel angle (rad); brakes: 4 torques (N m); throttle 0..1."""
        self.cmd[ph.C_DELTA] = steer
        self.cmd[ph.C_TB0 : ph.C_TB0 + 4] = brakes
        self.cmd[ph.C_THROTTLE] = throttle

    def step(self, duration: float):
        n = max(1, int(round(duration / PHYSICS_DT)))
        ph.step_physics(
            self.S, self.A, self.cmd, self.use_abs, self.use_esc, self._P,
            self.base_mu, self.patches, n, PHYSICS_DT, self.diag,
        )

    # ----- convenient read-outs
    @property
    def x(self):
        return self.S[ph.SX]

    @property
    def y(self):
        return self.S[ph.SY]

    @property
    def psi(self):
        return self.S[ph.SPSI]

    @property
    def vx(self):
        return self.S[ph.SVX]

    @property
    def vy(self):
        return self.S[ph.SVY]

    @property
    def yaw_rate(self):
        return self.S[ph.SR]

    @property
    def speed(self):
        return math.hypot(self.S[ph.SVX], self.S[ph.SVY])

    @property
    def sideslip(self):
        """Body sideslip angle (rad): how much the car is sliding sideways."""
        if abs(self.vx) < 0.5:
            return 0.0
        return math.atan2(self.vy, abs(self.vx))

    @property
    def world_velocity(self):
        c, s = math.cos(self.psi), math.sin(self.psi)
        return np.array([self.vx * c - self.vy * s, self.vx * s + self.vy * c])

    @property
    def steer(self):
        return self.A[ph.A_DELTA]

    @property
    def brake_torques(self):
        return self.A[ph.A_TB0 : ph.A_TB0 + 4].copy()

    @property
    def wheel_slip(self):
        return self.diag[ph.D_KAPPA : ph.D_KAPPA + 4].copy()

    @property
    def wheel_speeds(self):
        return self.S[ph.SW0 : ph.SW0 + 4] * self.p.wheel_radius

    @property
    def accel(self):
        return self.diag[ph.D_AX], self.diag[ph.D_AY]

    @property
    def mu_under_wheels(self):
        return self.diag[ph.D_MU : ph.D_MU + 4].copy()

    @property
    def esc_active(self):
        return self.diag[ph.D_ESC] > 0.5

    def pose(self):
        return self.x, self.y, self.psi
