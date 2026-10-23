"""Top-down animation of recorded episodes (one row per controller, time-synchronised)."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from crashai.geometry import box_corners  # noqa: E402
from crashai.params import VehicleParams  # noqa: E402
from crashai.world import LANE_WIDTH  # noqa: E402

COLORS = {
    "car": "#8a8f98", "truck": "#5b616b", "pedestrian": "#f28c28",
    "cyclist": "#9b59b6", "debris": "#8d6e63",
}
EGO_DRIVER = "#3b82f6"
EGO_AI = "#16a34a"
ROAD = "#3a3d42"


def _wheel_polys(x, y, psi, steer, p: VehicleParams):
    pos = [(p.a, p.track / 2), (p.a, -p.track / 2), (-p.b, p.track / 2), (-p.b, -p.track / 2)]
    c, s = math.cos(psi), math.sin(psi)
    out = []
    for i, (bx, by) in enumerate(pos):
        wx = x + bx * c - by * s
        wy = y + bx * s + by * c
        out.append(box_corners(wx, wy, psi + (steer if i < 2 else 0.0), 0.68, 0.26))
    return out


def draw_frame(ax, axb, frame, road, title, p: VehicleParams, view=(-15, 45)):
    ax.clear()
    axb.clear()
    ex, ey, epsi = frame["ego"]
    x0, x1 = ex + view[0], ex + view[1]
    ax.set_xlim(x0, x1)
    ax.set_ylim(road["y_min"] - 1.5, road["y_max"] + 1.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#cfd8cf")
    ax.set_xticks([])
    ax.set_yticks([])
    # road, patches, lanes, guardrails
    ax.add_patch(Rectangle((x0, road["y_min"]), x1 - x0, road["y_max"] - road["y_min"], color=ROAD, zorder=0))
    for px0, px1, py0, py1, mu in road["patches"]:
        py0, py1 = max(py0, road["y_min"]), min(py1, road["y_max"])
        ax.add_patch(Rectangle((px0, py0), px1 - px0, py1 - py0, color="#a5d8ff", alpha=0.45, zorder=1))
    for i in range(road["n_lanes"] - 1):
        yl = i * LANE_WIDTH + LANE_WIDTH / 2
        ax.plot([x0, x1], [yl, yl], color="white", lw=1, ls=(0, (6, 6)), zorder=1)
    for yg in (road["y_min"], road["y_max"]):
        ax.plot([x0, x1], [yg, yg], color="#e5e7eb", lw=3, zorder=1)
    # obstacles (truth) and perception
    for ox, oy, opsi, ol, ow, kind in frame["obstacles"]:
        ax.add_patch(Polygon(box_corners(ox, oy, opsi, ol, ow), color=COLORS[kind], zorder=3))
    for ox, oy, opsi, ol, ow, ghost in frame["perceived"]:
        ax.add_patch(Polygon(box_corners(ox, oy, opsi, ol, ow), fill=False, ls="--", lw=0.8,
                             ec="#ef4444" if ghost else "#22d3ee", zorder=4))
    # ego + wheels coloured by brake torque
    ai = frame["phase"] == "ai"
    ax.add_patch(Polygon(box_corners(ex, ey, epsi, p.length, p.width), color=EGO_AI if ai else EGO_DRIVER, zorder=5))
    for poly, b in zip(_wheel_polys(ex, ey, epsi, frame["steer"], p), frame["brakes"]):
        ax.add_patch(Polygon(poly, color=plt.cm.Reds(0.15 + 0.85 * b), ec="black", lw=0.4, zorder=6))
    ttc = frame["ttc"]
    status = "AI in control" if ai else "Driver"
    ax.set_title(f"{title}   |   t={frame['t']:.2f}s   {frame['speed'] * 3.6:5.1f} km/h   "
                 f"TTC={'—' if math.isinf(ttc) else f'{ttc:.1f}s'}   {status}"
                 + ("   ESC" if frame["esc"] else ""), fontsize=9, loc="left")
    if frame.get("end"):
        ax.text(0.5, 0.5, frame["end"], transform=ax.transAxes, ha="center", va="center", fontsize=18,
                color="white", weight="bold", bbox=dict(fc="#dc2626" if "CRASH" in frame["end"] or "GUARD" in frame["end"] else "#16a34a", ec="none"))
    # controls panel
    names = ["FL", "FR", "RL", "RR", "Thr", "Steer"]
    vals = list(frame["brakes"]) + [frame["throttle"], frame["steer"] / p.max_steer]
    cols = ["#dc2626"] * 4 + ["#16a34a", "#2563eb"]
    axb.barh(names[::-1], vals[::-1], color=cols[::-1])
    axb.set_xlim(-1, 1)
    axb.axvline(0, color="black", lw=0.6)
    axb.tick_params(labelsize=7)
    axb.set_xticks([-1, 0, 1])


def animate(episodes, out_path, fps=20, tail=1.0):
    """episodes: list of dicts with keys name, frames, road, outcome. Saves a GIF."""
    p = VehicleParams()
    n = len(episodes)
    dt = 0.05
    t_end = max(ep["frames"][-1]["t"] for ep in episodes) + tail
    t_start = min(ep["frames"][0]["t"] for ep in episodes)
    times = np.arange(t_start, t_end, dt)

    fig = plt.figure(figsize=(12, 2.6 * n))
    axes = []
    for i in range(n):
        top = 1 - (i + 1) / n
        ax = fig.add_axes([0.01, top + 0.04 / n, 0.80, 0.86 / n])
        axb = fig.add_axes([0.85, top + 0.12 / n, 0.13, 0.70 / n])
        axes.append((ax, axb))

    def frame_at(ep, t):
        fr = ep["frames"]
        idx = min(np.searchsorted([f["t"] for f in fr], t), len(fr) - 1)
        f = dict(fr[idx])
        if t > fr[-1]["t"]:
            f["end"] = ep["outcome_text"]
        return f

    def update(k):
        for (ax, axb), ep in zip(axes, episodes):
            draw_frame(ax, axb, frame_at(ep, times[k]), ep["road"], ep["name"], p)
        return []

    anim = FuncAnimation(fig, update, frames=len(times), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def road_dict(road):
    return {"y_min": road.y_min, "y_max": road.y_max, "n_lanes": road.n_lanes,
            "patches": [tuple(r) for r in road.patches]}


def outcome_text(info):
    o = info["outcome"]
    if o == "crash":
        return f"CRASH at {info['impact_speed'] * 3.6:.0f} km/h"
    if o == "guardrail":
        return f"GUARDRAIL at {info['impact_speed'] * 3.6:.0f} km/h"
    return {"stopped": "AVOIDED (stopped)", "handback": "AVOIDED (handed back)", "timeout": "TIMEOUT"}.get(o, o)
