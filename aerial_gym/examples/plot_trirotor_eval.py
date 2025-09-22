import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import re


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

    # Helper: sort columns by last integer occurrence
    def sort_cols_numeric(cols: List[str]) -> List[str]:
        def last_int(s: str) -> int:
            m = re.findall(r"(\d+)", s)
            return int(m[-1]) if m else -1
        return sorted(cols, key=last_int)

    # Collect column groups
    act_cols = sort_cols_numeric([c for c in seg.columns if re.match(r"^act_\d+$", c)])
    act_clipped_cols = sort_cols_numeric([c for c in seg.columns if re.match(r"^act_clipped_\d+$", c)])
    act_scaled_cols = sort_cols_numeric([c for c in seg.columns if re.match(r"^act_scaled_\d+$", c)])
    servo_next_rad_cols = sort_cols_numeric([c for c in seg.columns if re.match(r"^servo_next_\d+_rad$", c)])
    servo_next_deg_cols = sort_cols_numeric([c for c in seg.columns if re.match(r"^servo_next_\d+_deg$", c)])
    servo_cmd_last_cols = sort_cols_numeric([c for c in seg.columns if re.match(r"^servo_cmd_last_\d+_rad$", c)])
    motor_next_cols = sort_cols_numeric([c for c in seg.columns if re.match(r"^motor_next_\d+_N$", c)])
    motor_cmd_last_cols = sort_cols_numeric([c for c in seg.columns if re.match(r"^motor_cmd_last_\d+_N$", c)])
    motor_actual_last_cols = sort_cols_numeric([c for c in seg.columns if re.match(r"^motor_actual_last_\d+_N$", c)])

    n_act = len(act_cols)
    t = seg["t"] if "t" in seg.columns else seg.index

    # Build panels dynamically (each item: title, ylabel, list of (label, series))
    panels: List[Tuple[str, str, List[Tuple[str, pd.Series]]]] = []

    # Position
    if all(c in seg.columns for c in ["pos_x", "pos_y", "pos_z"]):
        panels.append((
            "Position vs Time",
            "Position [m]",
            [("x", seg["pos_x"]), ("y", seg["pos_y"]), ("z", seg["pos_z"])],
        ))

    # Attitude
    if all(c in seg.columns for c in ["roll", "pitch", "yaw"]):
        panels.append((
            "Attitude vs Time",
            "Attitude [deg]",
            [("roll", seg["roll"] * 180 / np.pi),
             ("pitch", seg["pitch"] * 180 / np.pi),
             ("yaw", seg["yaw"] * 180 / np.pi)],
        ))

    # Linear vel
    if all(c in seg.columns for c in ["vel_x", "vel_y", "vel_z"]):
        panels.append((
            "Linear Velocity vs Time",
            "Linear Velocity [m/s]",
            [("vx", seg["vel_x"]), ("vy", seg["vel_y"]), ("vz", seg["vel_z"])],
        ))

    # Angular vel
    if all(c in seg.columns for c in ["ang_x", "ang_y", "ang_z"]):
        panels.append((
            "Angular Velocity vs Time",
            "Angular Velocity [deg/s]",
            [("wx", seg["ang_x"] * 180 / np.pi),
             ("wy", seg["ang_y"] * 180 / np.pi),
             ("wz", seg["ang_z"] * 180 / np.pi)],
        ))

    # Motor actions (next step, scaled N)
    if motor_next_cols:
        panels.append((
            "Motor Actions vs Time (scaled next)",
            "Motor Thrust [N]",
            [(f"motor_{i}", seg[c]) for i, c in enumerate(motor_next_cols)],
        ))
    # Servo actions (next step, deg)
    if servo_next_deg_cols:
        panels.append((
            "Servo Actions vs Time (scaled next)",
            "Servo Angle [deg]",
            [(f"servo_{i}", seg[c]) for i, c in enumerate(servo_next_deg_cols)],
        ))

    # Raw mu actions
    if act_cols:
        # For 6D, split into servo/motor groups
        if len(act_cols) >= 6:
            panels.append((
                "Raw Mu vs Servo (0..2)",
                "mu (unitless)",
                [(f"servo_{i}", seg[act_cols[i]]) for i in range(3)],
            ))
            panels.append((
                "Raw Mu vs Motor (3..5)",
                "mu (unitless)",
                [(f"motor_{i}", seg[act_cols[3 + i]]) for i in range(3)],
            ))
        else:
            panels.append((
                "Raw Mu vs Motors",
                "mu (unitless)",
                [(f"motor_{i}", seg[c]) for i, c in enumerate(act_cols)],
            ))

    # Clipped actions [-1,1]
    if act_clipped_cols:
        if len(act_clipped_cols) >= 6:
            panels.append((
                "Clipped [-1,1] vs Servo (0..2)",
                "clipped",
                [(f"servo_{i}", -seg[act_clipped_cols[i]]) for i in range(3)],  # Multiply by -1 for x-axis reflection
            ))
            panels.append((
                "Clipped [-1,1] vs Motor (3..5)",
                "clipped",
                [(f"motor_{i}", seg[act_clipped_cols[3 + i]]) for i in range(3)],
            ))
        else:
            panels.append((
                "Clipped [-1,1] vs Motors",
                "clipped",
                [(f"motor_{i}", seg[c]) for i, c in enumerate(act_clipped_cols)],
            ))

    # Scaled actions (redundant with next-step panels but included for completeness)
    if act_scaled_cols:
        if len(act_scaled_cols) >= 6:
            panels.append((
                "Scaled vs Servo (deg)",
                "Servo [deg]",
                [(f"servo_{i}", seg[act_scaled_cols[i]] * 180 / np.pi) for i in range(3)],
            ))
            panels.append((
                "Scaled vs Motor (N)",
                "Motor [N]",
                [(f"motor_{i}", seg[act_scaled_cols[3 + i]]) for i in range(3)],
            ))
        else:
            panels.append((
                "Scaled vs Motors",
                "Motor [N]",
                [(f"motor_{i}", seg[c]) for i, c in enumerate(act_scaled_cols)],
            ))

    # Last-step commanded servo (convert rad->deg for plotting)
    if servo_cmd_last_cols:
        panels.append((
            "Servo Command (last) vs Target [deg]",
            "Servo [deg]",
            [(f"servo_{i}", seg[c] * 180 / np.pi) for i, c in enumerate(servo_cmd_last_cols)],
        ))

    # Last-step commanded vs actual motor thrusts
    if motor_cmd_last_cols or motor_actual_last_cols:
        series: List[Tuple[str, pd.Series]] = []
        for i, c in enumerate(motor_cmd_last_cols):
            series.append((f"cmd_{i}", seg[c]))
        for i, c in enumerate(motor_actual_last_cols):
            series.append((f"actual_{i}", seg[c]))
        panels.append((
            "Motor Thrust (last) vs Cmd vs Actual",
            "Motor [N]",
            series,
        ))

    # If nothing to plot, exit
    if not panels:
        print("No plottable columns found in CSV.")
        return

    # Create figure grid with 2 columns
    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(6, 3 * nrows)), sharex=True)
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = [axes]

    # Plot panels
    for ax, (title, ylabel, series) in zip(axes, panels):
        for label, y in series:
            ax.plot(t, y, label=label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if len(series) > 1:
            ax.legend(ncols=min(3, len(series)), fontsize=9)
        ax.set_xlabel("Time [s]")

    # Clear any unused axes
    for j in range(len(panels), len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"Trirotor Evaluation vs Episode {ep_id} (first {args.length} steps)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out = Path(args.save)
    fig.savefig(out, dpi=300)
    print(f"Saved figure to {out.resolve()}")


if __name__ == "__main__":
    main()
