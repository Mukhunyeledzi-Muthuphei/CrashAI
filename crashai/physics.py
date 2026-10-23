"""Compiled vehicle dynamics kernel.

A planar 4-wheel model:
  - body: x, y, heading, longitudinal/lateral velocity, yaw rate (3 degrees of freedom)
  - 4 wheel spin speeds
  - combined-slip Pacejka tires: braking/driving and cornering share one grip budget
  - quasi-static load transfer (front/rear under braking, left/right in corners)
  - FWD open differential with engine inertia (equal torque left/right)
  - actuator limits: steering rate, brake pressure build/release time
  - ABS slip limiter and a simple ESC yaw controller, both optional

Units are SI. Body frame: x forward, y left, yaw positive counter-clockwise.
Wheel order: FL, FR, RL, RR.
"""

import math

import numpy as np
from numba import njit

# Layout of the state array S
SX, SY, SPSI, SVX, SVY, SR = 0, 1, 2, 3, 4, 5
SW0 = 6  # wheel speeds 6..9
SAX, SAY = 10, 11  # filtered body accelerations (for load transfer)
STATE_SIZE = 12

# Layout of the actuator array A
A_DELTA = 0
A_TB0 = 1  # actual brake torques 1..4
ACT_SIZE = 5

# Layout of the command array
C_DELTA = 0
C_TB0 = 1  # brake torque requests 1..4
C_THROTTLE = 5
CMD_SIZE = 6

# Layout of the diagnostics array
D_KAPPA = 0  # 0..3 longitudinal slip
D_ALPHA = 4  # 4..7 slip angle (rad)
D_FZ = 8  # 8..11 vertical load
D_FX = 12  # 12..15 tire longitudinal force
D_FY = 16  # 16..19 tire lateral force
D_MU = 20  # 20..23 road friction under each wheel
D_AX, D_AY = 24, 25
D_ESC = 26
DIAG_SIZE = 27

G = 9.81


@njit(cache=True)
def magic_formula(s, B, C, E):
    bs = B * s
    return math.sin(C * math.atan(bs - E * (bs - math.atan(bs))))


@njit(cache=True)
def road_mu(x, y, base_mu, patches):
    """Friction at a point. patches rows: x0, x1, y0, y1, mu (later rows win)."""
    mu = base_mu
    for k in range(patches.shape[0]):
        if patches[k, 0] <= x <= patches[k, 1] and patches[k, 2] <= y <= patches[k, 3]:
            mu = patches[k, 4]
    return mu


@njit(cache=True)
def step_physics(S, A, cmd, use_abs, use_esc, P, base_mu, patches, n_sub, dt, diag):
    """Advance the car by n_sub substeps of dt seconds. Modifies S, A and diag in place."""
    m = P[0]
    Iz = P[1]
    a = P[2]
    b = P[3]
    tw = P[4]
    h = P[5]
    R = P[6]
    Iw = P[7]
    B = P[8]
    C = P[9]
    E = P[10]
    Tmax = P[11]
    Pmax = P[12]
    Je = P[13]
    TbF = P[14]
    TbR = P[15]
    t_build = P[16]
    t_rel = P[17]
    dmax = P[18]
    drate = P[19]
    kdrag = P[20]
    crr = P[21]
    abs_slip = P[22]
    t_dump = P[23]
    kesc = P[24]
    esc_db = P[25]
    Kus = P[26]
    L = a + b

    xi = np.array([a, a, -b, -b])
    yi = np.array([tw / 2, -tw / 2, tw / 2, -tw / 2])
    tbmax = np.array([TbF, TbF, TbR, TbR])

    delta_cmd = min(max(cmd[C_DELTA], -dmax), dmax)
    throttle = min(max(cmd[C_THROTTLE], 0.0), 1.0)

    tb_cmd = np.empty(4)
    Fz = np.empty(4)
    Fx = np.zeros(4)
    Fy = np.zeros(4)
    kap = np.zeros(4)
    alp = np.zeros(4)
    mu = np.empty(4)
    vden = np.empty(4)
    Ieff = np.empty(4)
    esc_active = 0.0
    axb = S[SAX]
    ayb = S[SAY]

    for _ in range(n_sub):
        x = S[SX]
        y = S[SY]
        psi = S[SPSI]
        vx = S[SVX]
        vy = S[SVY]
        r = S[SR]

        # --- steering actuator (rate limited)
        dd = delta_cmd - A[A_DELTA]
        lim = drate * dt
        if dd > lim:
            dd = lim
        elif dd < -lim:
            dd = -lim
        A[A_DELTA] += dd
        delta = A[A_DELTA]

        # --- road friction under each wheel
        cpsi = math.cos(psi)
        spsi = math.sin(psi)
        for i in range(4):
            wx = x + xi[i] * cpsi - yi[i] * spsi
            wy = y + xi[i] * spsi + yi[i] * cpsi
            mu[i] = road_mu(wx, wy, base_mu, patches)

        # --- brake requests, plus ESC
        for i in range(4):
            tb_cmd[i] = min(max(cmd[C_TB0 + i], 0.0), tbmax[i])
        if use_esc and vx > 3.0:
            mu_avg = 0.25 * (mu[0] + mu[1] + mu[2] + mu[3])
            r_des = vx * delta / (L + Kus * vx * vx)
            r_lim = 0.85 * mu_avg * G / vx
            if r_des > r_lim:
                r_des = r_lim
            elif r_des < -r_lim:
                r_des = -r_lim
            err = r - r_des
            if abs(err) > esc_db:
                extra = kesc * (abs(err) - esc_db)
                oversteer = abs(r) > abs(r_des)
                if err > 0.0:  # too much left yaw: brake the right side
                    w = 1 if oversteer else 3
                else:  # too much right yaw: brake the left side
                    w = 0 if oversteer else 2
                room = tbmax[w] - tb_cmd[w]
                tb_cmd[w] = min(tb_cmd[w] + extra, tbmax[w])
                if extra > room:
                    # wheel already at max: release the other wheel on the same axle instead
                    o = w ^ 1
                    tb_cmd[o] = max(0.0, tb_cmd[o] - (extra - room) * tbmax[o] / tbmax[w])
                esc_active = 1.0

        # --- vertical loads (quasi-static load transfer from filtered accelerations)
        Fzf = m * G * b / L / 2.0
        Fzr = m * G * a / L / 2.0
        dlong = m * S[SAX] * h / L / 2.0
        dlat_f = m * S[SAY] * h / tw * (b / L)
        dlat_r = m * S[SAY] * h / tw * (a / L)
        Fz[0] = Fzf - dlong - dlat_f
        Fz[1] = Fzf - dlong + dlat_f
        Fz[2] = Fzr + dlong - dlat_r
        Fz[3] = Fzr + dlong + dlat_r
        for i in range(4):
            if Fz[i] < 50.0:
                Fz[i] = 50.0

        # --- tire forces (combined slip: one grip budget per tire)
        cd = math.cos(delta)
        sd = math.sin(delta)
        for i in range(4):
            c = cd if i < 2 else 1.0
            s_ = sd if i < 2 else 0.0
            vxb = vx - r * yi[i]
            vyb = vy + r * xi[i]
            vl = vxb * c + vyb * s_
            vt = -vxb * s_ + vyb * c
            vd = max(abs(vl), 1.0)
            vden[i] = vd
            k = (S[SW0 + i] * R - vl) / vd
            if k < -1.0:
                k = -1.0
            elif k > 1.5:
                k = 1.5
            ta = vt / vd
            if ta < -3.0:
                ta = -3.0
            elif ta > 3.0:
                ta = 3.0
            ss = math.sqrt(k * k + ta * ta)
            if ss > 1e-9:
                F = mu[i] * Fz[i] * magic_formula(min(ss, 1.0), B, C, E)
                Fx[i] = F * k / ss
                Fy[i] = -F * ta / ss
            else:
                Fx[i] = 0.0
                Fy[i] = 0.0
            kap[i] = k
            alp[i] = math.atan(ta)
            # effective inertia for an implicit (stable) wheel-spin update
            Ieff[i] = Iw + dt * R * R * mu[i] * Fz[i] * B * C / vd

        # --- brake actuators with ABS
        for i in range(4):
            tb = A[A_TB0 + i]
            if use_abs and vx > 1.0 and kap[i] < -abs_slip:
                tb -= tbmax[i] / t_dump * dt  # dump pressure
            elif use_abs and vx > 1.0 and kap[i] < -0.6 * abs_slip and tb_cmd[i] > tb:
                pass  # hold pressure
            else:
                target = tb_cmd[i]
                if target > tb:
                    tb = min(target, tb + tbmax[i] / t_build * dt)
                else:
                    tb = max(target, tb - tbmax[i] / t_rel * dt)
            A[A_TB0 + i] = max(tb, 0.0)

        # --- open differential (front axle): equal drive torque to both front wheels
        wc = 0.5 * (S[SW0] + S[SW0 + 1])
        Te = throttle * min(Tmax, Pmax / max(wc, 1.0))
        load = 0.0
        for i in range(2):
            wv = S[SW0 + i]
            sgn = 1.0 if wv > 1e-6 else (-1.0 if wv < -1e-6 else 0.0)
            load += Fx[i] * R + A[A_TB0 + i] * sgn
        wcdot = (Te - load) / (Ieff[0] + Ieff[1] + Je)
        Td = Te - Je * wcdot

        # --- wheel spin update (implicit tire stiffness, brakes cannot reverse the wheel)
        for i in range(4):
            tdrv = 0.5 * Td if i < 2 else 0.0
            w_new = S[SW0 + i] + dt * (tdrv - Fx[i] * R) / Ieff[i]
            dec = dt * A[A_TB0 + i] / Ieff[i]
            if w_new > 0.0:
                w_new = max(0.0, w_new - dec)
            elif w_new < 0.0:
                w_new = min(0.0, w_new + dec)
            S[SW0 + i] = w_new

        # --- body dynamics
        sumFx = 0.0
        sumFy = 0.0
        Mz = 0.0
        for i in range(4):
            c = cd if i < 2 else 1.0
            s_ = sd if i < 2 else 0.0
            fxb = Fx[i] * c - Fy[i] * s_
            fyb = Fx[i] * s_ + Fy[i] * c
            sumFx += fxb
            sumFy += fyb
            Mz += xi[i] * fyb - yi[i] * fxb
        speed = math.sqrt(vx * vx + vy * vy)
        drag = kdrag * vx * abs(vx)
        rr = crr * m * G * vx / max(speed, 0.5)
        axb = (sumFx - drag - rr) / m
        ayb = sumFy / m
        S[SVX] = vx + dt * (axb + r * vy)
        S[SVY] = vy + dt * (ayb - r * vx)
        S[SR] = r + dt * Mz / Iz
        S[SX] = x + dt * (vx * cpsi - vy * spsi)
        S[SY] = y + dt * (vx * spsi + vy * cpsi)
        S[SPSI] = psi + dt * r
        f = dt / 0.05
        S[SAX] += f * (axb - S[SAX])
        S[SAY] += f * (ayb - S[SAY])

        # --- come to a clean stop
        if speed < 0.15 and throttle < 0.05:
            braking = A[A_TB0] + A[A_TB0 + 1] + A[A_TB0 + 2] + A[A_TB0 + 3]
            if braking > 200.0:
                S[SVX] = 0.0
                S[SVY] = 0.0
                S[SR] = 0.0
                for i in range(4):
                    S[SW0 + i] = 0.0

    for i in range(4):
        diag[D_KAPPA + i] = kap[i]
        diag[D_ALPHA + i] = alp[i]
        diag[D_FZ + i] = Fz[i]
        diag[D_FX + i] = Fx[i]
        diag[D_FY + i] = Fy[i]
        diag[D_MU + i] = mu[i]
    diag[D_AX] = axb
    diag[D_AY] = ayb
    diag[D_ESC] = esc_active
