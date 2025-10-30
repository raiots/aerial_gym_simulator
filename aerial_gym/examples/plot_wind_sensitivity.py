"""Plot wind-sensitivity diagnostics using academic figure styles.

This script complements ``offline_wind_sensitivity.py`` by visualising how the
learned policy reacts to wind-speed variations. It recreates the styling logic
from ``plot_trirotor_eval.py`` (SciencePlots/seaborn fallback) so the output is
suitable for papers or presentations.

The figure can include up to four panels:

1. Wind speed vs. time for a selected episode segment.
2. Binned mean servo commands (deg) vs. wind speed.
3. Binned mean motor thrusts (N) vs. wind speed.
4. Optional offline sweep curves (mean action vs. wind) when a sweep CSV is
   provided (typically generated via ``offline_wind_sensitivity.py --sweep-out``).

Example usage::

    python aerial_gym/examples/plot_wind_sensitivity.py \
        --csv aerial_gym/examples/trirotor_eval_6d.csv \
        --length 600 --episode 0 \
        --sweep airtmp/wind_sweep.csv \
        --save wind_sensitivity.pdf
"""

import argparse
import random
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def try_set_styles() -> None:
    """Apply seaborn/scienceplot styles when available, otherwise fallback."""

    try:
        import seaborn as sns  # noqa: F401

        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    try:
        import scienceplots  # noqa: F401

        try:
            plt.style.use(["science", "no-latex", "grid"])
        except Exception:
            plt.style.use(["science", "grid"])
    except Exception:
        pass


def pick_episode(df: pd.DataFrame, length: int, episode: Optional[int]) -> Tuple[int, pd.DataFrame]:
    """Select an episode with at least ``length`` steps and return a slice."""

    if "episode" not in df.columns:
        raise ValueError("CSV missing 'episode' column. Re-run eval_trirotor_csv.py with updated version.")

    groups = df.groupby("episode")
    candidates = [eid for eid, g in groups if len(g) >= length]
    if not candidates:
        raise ValueError(f"No episode with at least {length} steps found.")

    ep = episode if (episode is not None and episode in candidates) else random.choice(candidates)
    seg = groups.get_group(ep).reset_index(drop=True)
    return ep, seg.iloc[:length].copy()


def sort_cols_numeric(columns: Sequence[str], prefix: str) -> List[str]:
    pattern = re.compile(rf"^{prefix}(\d+)(?:.*)?$")

    def last_int(col: str) -> int:
        match = pattern.match(col)
        return int(match.group(1)) if match else -1

    return sorted(columns, key=last_int)


def make_bins(series: pd.Series, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(series.min(), series.max(), bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    return edges, mids


def aggregate_by_wind(
    wind: pd.Series, values: pd.DataFrame, bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges, mids = make_bins(wind, bins)
    bin_ids = pd.cut(wind, bins=edges, include_lowest=True, labels=False).to_numpy()
    n_bins = len(mids)
    mean = np.full((n_bins, values.shape[1]), np.nan, dtype=np.float64)
    std = np.full_like(mean, np.nan)
    val_np = values.to_numpy()
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        sel = val_np[mask]
        mean[b] = sel.mean(axis=0)
        std[b] = sel.std(axis=0, ddof=0)
    valid = ~np.isnan(mean).any(axis=1)
    return mids[valid], mean[valid], std[valid]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot wind sensitivity diagnostics")
    parser.add_argument("--csv", type=str, required=True, help="Evaluation CSV produced by eval_trirotor_csv.py")
    parser.add_argument("--length", type=int, default=600, help="Episode segment length for time-series panel")
    parser.add_argument("--episode", type=int, default=None, help="Specific episode id; defaults to random")
    parser.add_argument("--bins", type=int, default=16, help="Number of wind bins for action aggregation")
    parser.add_argument("--sweep", type=str, default=None, help="Optional sweep CSV from offline_wind_sensitivity")
    parser.add_argument("--save", type=str, default="wind_sensitivity.pdf", help="Output figure path")
    parser.add_argument("--show", action="store_true", help="Display figure window in addition to saving")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try_set_styles()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if "wind_speed" not in df.columns:
        raise ValueError("CSV does not contain 'wind_speed'; update eval_trirotor_csv.py and re-run evaluation.")

    ep_id, seg = pick_episode(df, args.length, args.episode)
    time_axis = seg["t"] if "t" in seg.columns else seg.index

    # Identify action columns
    act_scaled_cols = sort_cols_numeric([c for c in df.columns if c.startswith("act_scaled_")], "act_scaled_")
    if len(act_scaled_cols) < 6:
        raise ValueError("Expected at least 6 act_scaled columns for trirotor setup.")

    servo_cols = act_scaled_cols[:3]
    motor_cols = act_scaled_cols[3:6]

    wind_series = df["wind_speed"].astype(float)
    servo_values = df[servo_cols]
    motor_values = df[motor_cols]

    mids_servo, servo_mean, servo_std = aggregate_by_wind(wind_series, servo_values, args.bins)
    mids_motor, motor_mean, motor_std = aggregate_by_wind(wind_series, motor_values, args.bins)

    n_panels = 3
    sweep_df = None
    if args.sweep:
        sweep_path = Path(args.sweep)
        if not sweep_path.exists():
            raise FileNotFoundError(sweep_path)
        sweep_df = pd.read_csv(sweep_path)
        n_panels += 1

    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 2.5 * n_panels), sharex=False)
    if n_panels == 1:
        axes = [axes]

    ax0 = axes[0]
    ax0.plot(time_axis, seg["wind_speed"], color="tab:blue", linewidth=1.6)
    ax0.set_title(f"Episode {ep_id} - Wind Speed vs Time")
    ax0.set_ylabel("Wind [m/s]")
    ax0.set_xlabel("Time [s]" if "t" in seg.columns else "Step")

    ax1 = axes[1]
    servo_deg = np.rad2deg(servo_mean)
    servo_err = np.rad2deg(servo_std)
    for i in range(servo_mean.shape[1]):
        ax1.plot(mids_servo, servo_deg[:, i], label=f"Servo {i}", linewidth=1.8)
        ax1.fill_between(mids_servo, servo_deg[:, i] - servo_err[:, i], servo_deg[:, i] + servo_err[:, i], alpha=0.15)
    ax1.set_title("Binned Servo Command vs Wind")
    ax1.set_ylabel("Servo [deg]")
    ax1.set_xlabel("Wind [m/s]")
    ax1.legend(loc="best")

    ax2 = axes[2]
    for i in range(motor_mean.shape[1]):
        ax2.plot(mids_motor, motor_mean[:, i], label=f"Motor {i}", linewidth=1.8)
        ax2.fill_between(mids_motor, motor_mean[:, i] - motor_std[:, i], motor_mean[:, i] + motor_std[:, i], alpha=0.15)
    ax2.set_title("Binned Motor Thrust vs Wind")
    ax2.set_ylabel("Thrust [N]")
    ax2.set_xlabel("Wind [m/s]")
    ax2.legend(loc="best")

    if sweep_df is not None:
        ax3 = axes[3]
        wind_vals = sweep_df["wind"].to_numpy()
        act_cols = [c for c in sweep_df.columns if c.startswith("act_mean_")]
        for col in act_cols:
            idx = int(col.split("_")[-1])
            label = f"Act {idx}"
            ax3.plot(wind_vals, sweep_df[col], label=label, linewidth=1.6)
        ax3.set_title("Offline Sweep: Mean Action vs Wind")
        ax3.set_ylabel("Action (scaled)")
        ax3.set_xlabel("Wind [m/s]")
        ax3.legend(loc="best")

    fig.tight_layout()
    out_path = Path(args.save)
    fig.savefig(out_path, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
