import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def try_set_styles():
    # Seaborn
    try:
        import seaborn as sns  # noqa: F401
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    # SciencePlots (academic style)
    try:
        import scienceplots  # noqa: F401
        try:
            plt.style.use(["science", "no-latex", "grid"])
        except Exception:
            plt.style.use(["science", "grid"])
    except Exception:
        # Fallback: keep seaborn style if available
        pass


def pick_episode(df: pd.DataFrame, length: int, episode: Optional[int]) -> Tuple[int, pd.DataFrame]:
    if "episode" not in df.columns:
        raise ValueError("CSV missing 'episode' column. Re-run eval_trirotor_csv.py with updated version.")
    groups = df.groupby("episode")
    candidates = [eid for eid, g in groups if len(g) >= length]
    if not candidates:
        raise ValueError(f"No episode with at least {length} steps found.")
    if episode is not None and episode in candidates:
        ep = episode
    else:
        ep = random.choice(candidates)
    g = groups.get_group(ep).reset_index(drop=True)
    return ep, g.iloc[:length].copy()


def parse_args():
    p = argparse.ArgumentParser(description="Plot Trirotor CSV evaluation results (academic style)")
    p.add_argument("--csv", type=str, required=True, help="CSV file from eval_trirotor_csv.py")
    p.add_argument("--length", type=int, default=600, help="Episode segment length to plot")
    p.add_argument("--episode", type=int, default=None, help="Specific episode id to plot (default: random with required length)")
    p.add_argument("--save", type=str, default="trirotor_eval_plot.pdf", help="Output figure path (pdf/png)")
    return p.parse_args()


def main():
    args = parse_args()
    try_set_styles()

    df = pd.read_csv(args.csv)
    ep_id, seg = pick_episode(df, args.length, args.episode)

    # Detect available action columns
    act_cols = [c for c in seg.columns if c.startswith("act_")]
    n_act = len(act_cols)

    t = seg["t"] if "t" in seg.columns else seg.index

    # Figure layout
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

    # 1) Position
    if all(c in seg.columns for c in ["pos_x", "pos_y", "pos_z"]):
        axes[0].plot(t, seg["pos_x"], label="x")
        axes[0].plot(t, seg["pos_y"], label="y")
        axes[0].plot(t, seg["pos_z"], label="z")
        axes[0].set_ylabel("Position [m]")
        axes[0].set_title("Position vs Time")
        axes[0].legend(ncols=3, fontsize=9)
    else:
        axes[0].text(0.5, 0.5, "Position not in CSV", ha="center", va="center")

    # 2) Attitude (roll, pitch, yaw)
    if all(c in seg.columns for c in ["roll", "pitch", "yaw"]):
        axes[1].plot(t, seg["roll"], label="roll")
        axes[1].plot(t, seg["pitch"], label="pitch")
        axes[1].plot(t, seg["yaw"], label="yaw")
        axes[1].set_ylabel("Attitude [rad]")
        axes[1].set_title("Attitude vs Time")
        axes[1].legend(ncols=3, fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "Attitude not in CSV", ha="center", va="center")

    # 3) Linear velocity
    if all(c in seg.columns for c in ["vel_x", "vel_y", "vel_z"]):
        axes[2].plot(t, seg["vel_x"], label="vx")
        axes[2].plot(t, seg["vel_y"], label="vy")
        axes[2].plot(t, seg["vel_z"], label="vz")
        axes[2].set_ylabel("Linear Velocity [m/s]")
        axes[2].set_title("Linear Velocity vs Time")
        axes[2].legend(ncols=3, fontsize=9)
    else:
        axes[2].text(0.5, 0.5, "Linear velocity not in CSV", ha="center", va="center")

    # 4) Angular velocity
    if all(c in seg.columns for c in ["ang_x", "ang_y", "ang_z"]):
        axes[3].plot(t, seg["ang_x"], label="wx")
        axes[3].plot(t, seg["ang_y"], label="wy")
        axes[3].plot(t, seg["ang_z"], label="wz")
        axes[3].set_ylabel("Angular Velocity [rad/s]")
        axes[3].set_title("Angular Velocity vs Time")
        axes[3].legend(ncols=3, fontsize=9)
    else:
        axes[3].text(0.5, 0.5, "Angular velocity not in CSV", ha="center", va="center")

    # 5) Motor actions
    if n_act >= 3:
        # If we have 6 actions, last 3 are motor thrusts; if only 3, they are all motor
        if n_act >= 6:
            motor_cols = act_cols[3:6]
        else:
            motor_cols = act_cols
        for k, c in enumerate(motor_cols):
            axes[4].plot(t, seg[c], label=f"motor_{k}")
        axes[4].set_ylabel("Motor Thrust [N]")
        axes[4].set_title("Motor Actions vs Time")
        axes[4].legend(ncols=3, fontsize=9)
    else:
        axes[4].text(0.5, 0.5, "Motor actions not in CSV", ha="center", va="center")

    # 6) Servo actions (first 3 actions for 6D case)
    if n_act >= 6:
        for k, c in enumerate(act_cols[0:3]):
            axes[5].plot(t, seg[c], label=f"servo_{k}")
        axes[5].set_ylabel("Servo Angle [rad]")
        axes[5].set_title("Servo Actions vs Time")
        axes[5].legend(ncols=3, fontsize=9)
    else:
        axes[5].text(0.5, 0.5, "Servo actions not available (need 6D)", ha="center", va="center")

    for ax in axes:
        ax.set_xlabel("Time [s]")

    fig.suptitle(f"Trirotor Evaluation — Episode {ep_id} (first {args.length} steps)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out = Path(args.save)
    fig.savefig(out, dpi=300)
    print(f"Saved figure to {out.resolve()}")


if __name__ == "__main__":
    main()
