"""Plots a `gnoci run --record` log: servo voltage, joint movement and tilt.

    uv run python scripts/plot_power_log.py logs/power/run1.npz [--zoom 38 42]

Saves a .png next to the log. Steps that took over 3 control periods (a servo
read that timed out and was tried again) are marked on every panel.
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

JOINTS = ["hip_lat", "hip", "knee"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # one per joint, same on both legs
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
MARK = "#b0aea5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--zoom", nargs=2, type=float, metavar=("START", "END"),
                        help="Only plot this time window, in seconds.")
    args = parser.parse_args()

    d = np.load(args.log)
    t, obs, voltage = d["t"], d["obs"], d["voltage"]
    pos, vel = obs[:, 9:15], obs[:, 15:21]
    tilt = np.degrees(np.arccos(np.clip(-d["gravity"][:, 2], -1, 1)))
    period = 1 / float(d["control_freq"])
    slow = t[1:][np.diff(t) > 3 * period]  # step that ended late

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "lines.linewidth": 1.2,
    })
    fig, axes = plt.subplots(6, 1, figsize=(13, 13), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1, 1, 1, 1, 0.8]})

    ax = axes[0]
    ax.fill_between(t, voltage.min(1), voltage.max(1), color=COLORS[0], alpha=0.25,
                    linewidth=0, label="lowest to highest servo")
    ax.plot(t, voltage.mean(1), color=COLORS[0], label="mean of 6 servos")
    ax.set_ylabel("Supply voltage (V)")
    ax.set_title("Servo supply voltage", loc="left", color=INK)

    for row, (side, offset) in enumerate((("Left", 0), ("Right", 3)), start=1):
        for k, name in enumerate(JOINTS):
            axes[row].plot(t, pos[:, offset + k], color=COLORS[k], label=name)
            axes[row + 2].plot(t, vel[:, offset + k], color=COLORS[k], label=name,
                               linewidth=0.9)
        axes[row].set_ylabel("Angle (rad)")
        axes[row].set_title(f"{side} leg joint angles (from home)", loc="left", color=INK)
        axes[row + 2].set_ylabel("Speed (rad/s)")
        axes[row + 2].set_title(f"{side} leg joint velocities", loc="left", color=INK)

    ax = axes[5]
    ax.plot(t, tilt, color=INK)
    ax.set_ylabel("Tilt (°)")
    ax.set_title("Torso tilt from upright (filter)", loc="left", color=INK)
    ax.set_xlabel("Time (s)")

    for ax in axes:
        for s in slow:
            ax.axvline(s, color=MARK, linewidth=0.8, zorder=0)
    for ax in axes[1:5]:
        ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=8)
    axes[0].plot([], [], color=MARK, label="step over 60 ms (servo read retried)")
    axes[0].legend(loc="lower left", frameon=False, ncol=3, fontsize=8)

    if args.zoom:
        axes[-1].set_xlim(*args.zoom)
        # Rescale each panel's y to the visible window.
        m = (t >= args.zoom[0]) & (t <= args.zoom[1])
        for ax, data in zip(axes, [voltage, pos[:, :3], pos[:, 3:], vel[:, :3], vel[:, 3:], tilt]):
            lo, hi = np.nanmin(data[m]), np.nanmax(data[m])
            pad = 0.08 * (hi - lo or 1)
            ax.set_ylim(lo - pad, hi + pad)
    else:
        axes[-1].set_xlim(t[0], t[-1])

    fig.suptitle(f"{args.log.name}: {d['model']}, {len(slow)} slow steps",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    suffix = f"_{args.zoom[0]:g}-{args.zoom[1]:g}s" if args.zoom else ""
    out = args.log.with_name(args.log.stem + suffix + ".png")
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
