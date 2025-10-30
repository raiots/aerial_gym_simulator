from __future__ import annotations

"""Unified wind-sensitivity interpretability pipeline for the trirotor policy.

This script performs three stages in one pass:

1. Load evaluation CSV data, retain only episodes with sufficient length, and
   extract the last N steps (steady-flight window) for analysis.
2. Reconstruct observation tensors, load the RL-Games actor, and compute
   gradients of each action with respect to wind speed as well as hidden-layer
   activations for interpretability.
3. Produce academic-style visualisations comparing wind speed, servo commands,
   action sensitivities, and the behaviour of the most wind-correlated hidden
   units.

Example usage::

    python aerial_gym/examples/wind_sensitivity_analysis.py \
        --csv aerial_gym/examples/trirotor_eval_6d.csv \
        --checkpoint runs/.../nn/trirotor_6d.pth \
        --actor-activation relu \
        --output airtmp/wind_interpretability.pdf
"""

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility loaders (avoid importing aerial_gym to keep isaacgym optional)
# ---------------------------------------------------------------------------


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def load_custom_logger():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "logging.py"
    module = load_module(module_path, "logging_offline")
    return module.CustomLogger


def load_mlp_class():
    module_path = Path(__file__).resolve().parent / "rl_games_example" / "rl_games_inference.py"
    module = load_module(module_path, "rl_games_inference_offline")
    return module.MLP


CustomLogger = load_custom_logger()


# ---------------------------------------------------------------------------
# Styling utilities (reused from plot_trirotor_eval.py)
# ---------------------------------------------------------------------------


def try_set_styles() -> None:
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wind sensitivity interpretability analysis")
    parser.add_argument("--csv", type=str, required=True, help="Evaluation CSV from eval_trirotor_csv.py")
    parser.add_argument("--checkpoint", type=str, required=True, help="RL-Games actor checkpoint (.pth)")
    parser.add_argument("--actor-activation", type=str, default="relu", help="Activation when rebuilding actor")
    parser.add_argument("--min-episode-length", type=int, default=600, help="Minimum episode length to retain")
    parser.add_argument("--tail-steps", type=int, default=100, help="Number of terminal steps per episode to analyse")
    parser.add_argument("--bins", type=int, default=16, help="Binning for wind-conditioned statistics")
    parser.add_argument("--top-hidden", type=int, default=5, help="Number of hidden units to visualise per layer")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (auto-fallback if CUDA unavailable)")
    parser.add_argument("--output", type=str, default="wind_sensitivity_report.pdf", help="Result figure path")
    parser.add_argument(
        "--activation-out",
        type=str,
        default=None,
        help="Directory to store activation CSV/histograms (defaults to <output>_activations)",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for model evaluation/gradients")
    parser.add_argument(
        "--activation-layers",
        type=str,
        default="act1,act2",
        help="Comma-separated module names for activation plots (empty = all)",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            f"CUDA device '{device_str}' requested but unavailable; falling back to CPU"
        )
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def sort_cols_numeric(columns: Sequence[str], prefix: str) -> List[str]:
    import re

    pattern = re.compile(rf"^{prefix}(\d+)(?:.*)?$")

    def last_int(col: str) -> int:
        match = pattern.match(col)
        return int(match.group(1)) if match else -1

    return sorted(columns, key=last_int)


def filter_stable_segments(df: pd.DataFrame, min_len: int, tail_steps: int) -> pd.DataFrame:
    segments = []
    for episode, group in df.groupby("episode"):
        if len(group) < min_len:
            continue
        tail = group.tail(tail_steps).copy()
        tail["stable_step"] = np.arange(len(tail))
        if "t" in tail.columns:
            tail["t_rel"] = tail["t"] - tail["t"].iloc[0]
        else:
            tail["t_rel"] = tail["stable_step"]
        segments.append(tail)
    if not segments:
        raise ValueError(
            f"No episodes met requirements (min_len={min_len}). Re-run eval or adjust thresholds."
        )
    return pd.concat(segments, ignore_index=True)


def detect_wind_index(obs_matrix: np.ndarray, wind_column: np.ndarray) -> int:
    diffs = np.abs(obs_matrix - wind_column[:, None])
    mean_diff = np.nanmean(diffs, axis=0)
    return int(np.nanargmin(mean_diff))


# ---------------------------------------------------------------------------
# Model analysis helpers
# ---------------------------------------------------------------------------


def register_activation_hooks(model: nn.Module) -> Tuple[Dict[str, List[torch.Tensor]], List[torch.utils.hooks.RemovableHandle]]:
    activations: Dict[str, List[torch.Tensor]] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def hook_fn(name: str):
        def fn(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                activations.setdefault(name, []).append(output.detach().cpu())

        return fn

    interesting_types = (nn.Linear, nn.ELU, nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU, nn.SiLU, nn.GELU, nn.Identity)
    for name, module in model.network.named_modules():  # type: ignore[attr-defined]
        if isinstance(module, interesting_types):
            handles.append(module.register_forward_hook(hook_fn(name)))

    return activations, handles


def compute_gradients_and_activations(
    model: torch.nn.Module,
    observations: np.ndarray,
    wind_idx: int,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    model = model.to(device).eval()

    act_buffers, handles = register_activation_hooks(model)

    grads_list: List[torch.Tensor] = []
    actions_list: List[torch.Tensor] = []

    obs_tensor = torch.from_numpy(observations)
    num_samples = obs_tensor.shape[0]

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch = obs_tensor[start:end].to(device=device, dtype=torch.float32)
        batch.requires_grad_(True)

        outputs = model(batch)
        actions_list.append(outputs.detach().cpu())

        grad_components: List[torch.Tensor] = []
        for idx in range(outputs.shape[1]):
            grad = torch.autograd.grad(outputs[:, idx].sum(), batch, retain_graph=True)[0][:, wind_idx]
            grad_components.append(grad.detach().cpu())
        grads_batch = torch.stack(grad_components, dim=1)
        grads_list.append(grads_batch)

    for handle in handles:
        handle.remove()

    actions = torch.cat(actions_list, dim=0).numpy()
    grads = torch.cat(grads_list, dim=0).numpy()

    activations_np: Dict[str, np.ndarray] = {}
    for name, tensors in act_buffers.items():
        if tensors:
            activations_np[name] = torch.cat(tensors, dim=0).numpy()

    return actions, grads, activations_np


def compute_correlations(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = reference - reference.mean()
    ref_ss = np.sum(ref * ref)
    vals = values - values.mean(axis=0, keepdims=True)
    numerator = np.sum(vals * ref[:, None], axis=0)
    denom = np.sqrt(np.sum(vals * vals, axis=0) * ref_ss) + 1e-8
    return numerator / denom


def aggregate_by_wind(
    wind: np.ndarray, values: np.ndarray, bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(wind.min(), wind.max(), bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    bin_ids = np.digitize(wind, edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, len(mids) - 1)

    mean = np.full((len(mids), values.shape[1]), np.nan, dtype=np.float64)
    std = np.full_like(mean, np.nan)

    for b in range(len(mids)):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        subset = values[mask]
        mean[b] = subset.mean(axis=0)
        std[b] = subset.std(axis=0, ddof=0)

    valid = ~np.isnan(mean).any(axis=1)
    return mids[valid], mean[valid], std[valid]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_analysis(
    output_path: Path,
    segments: pd.DataFrame,
    wind: np.ndarray,
    servo_deg: np.ndarray,
    motor_thrust: np.ndarray,
    grads: np.ndarray,
    top_hidden_profiles: Dict[str, Dict[str, np.ndarray]],
    bins: int,
) -> None:
    try_set_styles()

    fig, axes = plt.subplots(5, 1, figsize=(7.0, 16.5), sharex=False)

    # Panel 1: wind speed trajectories during steady window
    ax0 = axes[0]
    for episode, group in segments.groupby("episode"):
        ax0.plot(group["t_rel"], group["wind_speed"], linewidth=1.2, alpha=0.6, label=f"ep {episode}")
    ax0.set_title("Wind speed during steady-flight window")
    ax0.set_ylabel("Wind [m/s]")
    ax0.set_xlabel("Relative time [s]")
    ax0.legend(loc="best", fontsize=8)

    # Panel 2: servo angles vs wind
    mids_servo, mean_servo, std_servo = aggregate_by_wind(wind, servo_deg, bins)
    ax1 = axes[1]
    for idx in range(mean_servo.shape[1]):
        ax1.plot(mids_servo, mean_servo[:, idx], linewidth=1.8, label=f"Servo {idx}")
        ax1.fill_between(
            mids_servo,
            mean_servo[:, idx] - std_servo[:, idx],
            mean_servo[:, idx] + std_servo[:, idx],
            alpha=0.15,
        )
    ax1.set_title("Servo command vs wind (last steps of qualifying episodes)")
    ax1.set_ylabel("Servo [deg]")
    ax1.set_xlabel("Wind [m/s]")
    ax1.legend(loc="best")

    # Panel 3: motor thrust vs wind
    mids_motor, mean_motor, std_motor = aggregate_by_wind(wind, motor_thrust, bins)
    ax2 = axes[2]
    for idx in range(mean_motor.shape[1]):
        ax2.plot(mids_motor, mean_motor[:, idx], linewidth=1.8, label=f"Motor {idx}")
        ax2.fill_between(
            mids_motor,
            mean_motor[:, idx] - std_motor[:, idx],
            mean_motor[:, idx] + std_motor[:, idx],
            alpha=0.15,
        )
    ax2.set_title("Motor thrust vs wind")
    ax2.set_ylabel("Thrust [N]")
    ax2.set_xlabel("Wind [m/s]")
    ax2.legend(loc="best")

    # Panel 4: action gradient sensitivity
    mids_grad, mean_grad, std_grad = aggregate_by_wind(wind, grads, bins)
    ax3 = axes[3]
    for idx in range(mean_grad.shape[1]):
        ax3.plot(mids_grad, mean_grad[:, idx], linewidth=1.6, label=f"grad a{idx}")
        ax3.fill_between(
            mids_grad,
            mean_grad[:, idx] - std_grad[:, idx],
            mean_grad[:, idx] + std_grad[:, idx],
            alpha=0.12,
        )
    ax3.set_title("Action sensitivity (d action / d wind)")
    ax3.set_ylabel("Gradient")
    ax3.set_xlabel("Wind [m/s]")
    ax3.legend(loc="best")

    # Panel 5: hidden-unit responses
    ax4 = axes[4]
    if top_hidden_profiles:
        for layer_name, payload in top_hidden_profiles.items():
            mids = payload["wind"]
            mean = payload["mean"]
            std = payload["std"]
            labels = payload["labels"]
            for idx in range(mean.shape[1]):
                ax4.plot(mids, mean[:, idx], linewidth=1.5, label=f"{layer_name} {labels[idx]}")
                ax4.fill_between(
                    mids, mean[:, idx] - std[:, idx], mean[:, idx] + std[:, idx], alpha=0.10
                )
        ax4.set_title("Top wind-correlated hidden activations")
        ax4.set_ylabel("Activation")
        ax4.set_xlabel("Wind [m/s]")
        ax4.legend(loc="best", fontsize=8)
    else:
        ax4.set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.45)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def main() -> None:
    global logger

    args = parse_args()
    csv_path = Path(args.csv)
    ckpt_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    logger = CustomLogger("wind_sensitivity")

    df = pd.read_csv(csv_path)
    if "wind_speed" not in df.columns:
        raise ValueError("CSV is missing 'wind_speed'. Re-run eval_trirotor_csv.py with latest version.")

    segments = filter_stable_segments(df, args.min_episode_length, args.tail_steps)
    logger.info(
        f"Retained {segments['episode'].nunique()} episodes "
        f"({len(segments)} samples) for steady-flight analysis."
    )

    obs_cols = sort_cols_numeric([c for c in segments.columns if c.startswith("obs_")], "obs_")
    if not obs_cols:
        raise ValueError("CSV lacks obs_* columns; update evaluation logging.")

    obs_matrix = segments[obs_cols].to_numpy(dtype=np.float32)
    wind_column = segments["wind_speed"].to_numpy(dtype=np.float32)
    wind_idx = detect_wind_index(obs_matrix, wind_column)
    logger.info(f"Detected wind observation index: obs_{wind_idx}")

    device = resolve_device(args.device)
    MLP = load_mlp_class()
    model = MLP(obs_matrix.shape[1], None, str(ckpt_path), activation=args.actor_activation)

    _, grads, activations = compute_gradients_and_activations(
        model, obs_matrix, wind_idx, device, args.batch_size
    )

    grad_mean = grads.mean(axis=0)
    grad_std = grads.std(axis=0, ddof=0)
    for idx, (mean_val, std_val) in enumerate(zip(grad_mean, grad_std)):
        logger.info(
            f"Gradient stats (action {idx} vs wind): mean={mean_val:.6f}, std={std_val:.6f}"
        )

    activation_out_dir = (
        Path(args.activation_out)
        if args.activation_out
        else output_path.with_name(output_path.stem + "_activations")
    )
    ensure_dir(activation_out_dir)

    requested_layers = [s.strip() for s in args.activation_layers.split(",") if s.strip()]
    if not requested_layers:
        requested_layers = list(activations.keys())

    top_hidden_profiles: Dict[str, Dict[str, np.ndarray]] = {}
    for layer_name, activ in activations.items():
        if layer_name not in requested_layers:
            continue
        corr = compute_correlations(activ, wind_column)
        order = np.argsort(-np.abs(corr))[: args.top_hidden]
        labels = [f"unit {i} (r={corr[i]:.2f})" for i in order]
        mids, mean_vals, std_vals = aggregate_by_wind(wind_column, activ[:, order], args.bins)
        top_hidden_profiles[layer_name] = {
            "wind": mids,
            "mean": mean_vals,
            "std": std_vals,
            "labels": labels,
        }
        logger.info(
            f"Layer {layer_name}: top correlations = {', '.join(labels)}"
        )

        pd.DataFrame(activ).to_csv(
            activation_out_dir / f"activations_{layer_name.replace('.', '_')}.csv", index=False
        )

        for unit_idx in order:
            plt.figure(figsize=(5, 3.2))
            plt.hist(activ[:, unit_idx], bins=60, color="tab:blue", alpha=0.8)
            plt.title(f"Activation histogram — {layer_name} unit {unit_idx}")
            plt.xlabel("Activation")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(activation_out_dir / f"act_hist_{layer_name}_unit{unit_idx}.pdf")
            plt.close()

    act_scaled_cols = sort_cols_numeric([c for c in segments.columns if c.startswith("act_scaled_")], "act_scaled_")
    if len(act_scaled_cols) < 6:
        raise ValueError("Expected at least 6 act_scaled columns for servo/motor actions.")

    servo_rad = segments[act_scaled_cols[:3]].to_numpy(dtype=np.float32)
    servo_deg = np.rad2deg(servo_rad)
    motor_thrust = segments[act_scaled_cols[3:6]].to_numpy(dtype=np.float32)

    plot_analysis(
        output_path,
        segments,
        wind_column,
        servo_deg,
        motor_thrust,
        grads,
        top_hidden_profiles,
        args.bins,
    )

    logger.info(f"Saved interpretability figure to {output_path}")


if __name__ == "__main__":
    main()
