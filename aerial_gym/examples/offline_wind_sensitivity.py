from __future__ import annotations

"""Offline policy sensitivity analysis for wind disturbances.

This script operates purely on previously logged CSV data and a trained
RL-Games actor checkpoint. It estimates how sensitive each action output is to
the wind-speed observation by:

* reconstructing observation vectors from CSV rows (columns `obs_*`),
* locating the wind-speed observation either from the dedicated `wind_speed`
  column or by matching against observation entries,
* computing per-action gradients ``d action_i / d wind_speed`` via autograd,
* sweeping wind-speed values to visualise the policy response.

Example usage:

.. code-block:: bash

    python aerial_gym/examples/offline_wind_sensitivity.py \
        --csv aerial_gym/examples/trirotor_eval_6d.csv \
        --checkpoint runs/.../nn/trirotor_6d.pth \
        --actor-activation relu

The script prints summary statistics to stdout and writes an optional sweep
table to disk when ``--sweep-out`` is provided.
"""

import argparse
import importlib.util
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

def load_custom_logger():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "logging.py"
    spec = importlib.util.spec_from_file_location("logging_offline", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load logger module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module.CustomLogger


CustomLogger = load_custom_logger()
logger = CustomLogger("offline_wind_sensitivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline wind sensitivity study")
    parser.add_argument("--csv", type=str, required=True, help="Path to eval CSV with obs_* columns")
    parser.add_argument("--checkpoint", type=str, required=True, help="RL-Games actor checkpoint (.pth)")
    parser.add_argument("--actor-activation", type=str, default="relu", help="Activation to rebuild actor (e.g. relu)")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for inference")
    parser.add_argument("--max-samples", type=int, default=4096, help="Max rows to use from CSV for gradient stats")
    parser.add_argument("--wind-index", type=int, default=None, help="Observation index for wind (auto-detected if omitted)")
    parser.add_argument("--sweep-points", type=int, default=21, help="Points for wind sweep")
    parser.add_argument("--sweep-out", type=str, default=None, help="Optional CSV path to dump wind sweep table")
    parser.add_argument("--obs-clip", type=float, default=None, help="Optional clip for observations before inference")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for sample selection")
    return parser.parse_args()


def load_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
    if data.size == 0:
        raise RuntimeError(f"File {path} contains no data rows")
    names = list(data.dtype.names)
    arr = data.view((np.float32, len(data.dtype.names)))
    return arr, names


def extract_observations(table: np.ndarray, headers: Sequence[str]) -> tuple[np.ndarray, Optional[np.ndarray], list[int]]:
    obs_cols = [i for i, h in enumerate(headers) if h.startswith("obs_")]
    if not obs_cols:
        raise RuntimeError("CSV is missing obs_* columns. Re-run eval with latest script.")
    obs_matrix = table[:, obs_cols]
    wind = None
    if "wind_speed" in headers:
        wind = table[:, headers.index("wind_speed")]
    return obs_matrix, wind, obs_cols


def detect_wind_index(obs_matrix: np.ndarray, wind: Optional[np.ndarray], provided_idx: Optional[int]) -> int:
    if provided_idx is not None:
        if provided_idx < 0 or provided_idx >= obs_matrix.shape[1]:
            raise ValueError("wind_index out of bounds for observation dimension")
        return provided_idx
    if wind is None:
        raise RuntimeError("No wind index provided and CSV lacks wind_speed column")
    diffs = np.abs(obs_matrix - wind[:, None])
    idx = int(np.nanargmin(np.nanmean(diffs, axis=0)))
    logger.info(f"Auto-detected wind observation at obs_{idx}")
    return idx


def select_samples(obs_matrix: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    mask = ~np.isnan(obs_matrix).any(axis=1)
    filtered = obs_matrix[mask]
    if filtered.size == 0:
        raise RuntimeError("All observation rows contain NaNs; cannot proceed")
    rng = np.random.default_rng(seed)
    if filtered.shape[0] <= max_samples:
        return filtered
    idx = rng.choice(filtered.shape[0], size=max_samples, replace=False)
    return filtered[idx]


def compute_gradients(model: torch.nn.Module, obs_batch: torch.Tensor, wind_idx: int) -> torch.Tensor:
    obs_batch.requires_grad_(True)
    actions = model(obs_batch)
    grads = []
    for i in range(actions.shape[1]):
        grad_i = torch.autograd.grad(actions[:, i].sum(), obs_batch, retain_graph=True)[0][:, wind_idx]
        grads.append(grad_i)
    obs_batch.requires_grad_(False)
    return torch.stack(grads, dim=1)


def make_sweep(model: torch.nn.Module, base_obs: torch.Tensor, wind_idx: int, wind_vals: torch.Tensor) -> torch.Tensor:
    """Evaluate actor outputs across a range of wind inputs."""
    tiled = base_obs.unsqueeze(0).repeat(wind_vals.shape[0], 1, 1)
    tiled[:, :, wind_idx] = wind_vals.unsqueeze(1)
    flat = tiled.reshape(-1, base_obs.shape[1])
    with torch.no_grad():
        actions = model(flat)
    return actions.view(wind_vals.shape[0], base_obs.shape[0], -1)


def load_mlp_class():
    """Load RL-Games MLP without importing aerial_gym (avoids isaacgym dependency)."""

    module_path = Path(__file__).resolve().parent / "rl_games_example" / "rl_games_inference.py"
    spec = importlib.util.spec_from_file_location("rl_games_inference_offline", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module.MLP


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            f"CUDA device '{device_str}' requested but unavailable; falling back to CPU"
        )
        return torch.device("cpu")
    return torch.device(device_str)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    ckpt_path = Path(args.checkpoint)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    table, headers = load_csv(csv_path)
    obs_matrix, wind_column, obs_cols = extract_observations(table, headers)
    wind_idx = detect_wind_index(obs_matrix, wind_column, args.wind_index)

    samples = select_samples(obs_matrix, args.max_samples, args.seed)
    obs_dim = samples.shape[1]
    logger.info(f"Loaded {samples.shape[0]} observation samples (dim={obs_dim})")

    device = resolve_device(args.device)
    MLP_cls = load_mlp_class()
    model = MLP_cls(obs_dim, None, args.checkpoint, activation=args.actor_activation).to(device).eval()
    act_dim = model.network[-1].out_features  # type: ignore[attr-defined]
    logger.info(f"Actor action dim inferred as {act_dim}")

    obs_tensor = torch.from_numpy(samples).to(device=device, dtype=torch.float32)
    if args.obs_clip is not None:
        obs_tensor = torch.clamp(obs_tensor, -args.obs_clip, args.obs_clip)

    grad_tensor = compute_gradients(model, obs_tensor.clone(), wind_idx)
    grad_np = grad_tensor.detach().cpu().numpy()
    grad_mean = grad_np.mean(axis=0)
    grad_std = grad_np.std(axis=0)
    for i in range(act_dim):
        logger.info(f"grad(action_{i} vs wind): mean={grad_mean[i]:.6f}, std={grad_std[i]:.6f}")

    if args.sweep_out or wind_column is not None:
        if wind_column is not None:
            sweep_min = float(np.nanmin(wind_column))
            sweep_max = float(np.nanmax(wind_column))
        else:
            sweep_min, sweep_max = -1.0, 1.0
        wind_vals = torch.linspace(sweep_min, sweep_max, args.sweep_points, device=device)
        base_obs = obs_tensor[: min(16, obs_tensor.shape[0])]
        actions = make_sweep(model, base_obs, wind_idx, wind_vals)
        mean_actions = actions.mean(dim=1).cpu().numpy()
        logger.info("Wind sweep (mean action per wind value):")
        for idx, w in enumerate(wind_vals.cpu().numpy()):
            act_vals = ", ".join(f"{v:.4f}" for v in mean_actions[idx])
            logger.info(f"  wind={w:.4f} -> [{act_vals}]")

        if args.sweep_out:
            sweep_path = Path(args.sweep_out)
            sweep_path.parent.mkdir(parents=True, exist_ok=True)
            header = ",".join(["wind"] + [f"act_mean_{i}" for i in range(act_dim)])
            body_lines = [
                ",".join(
                    [f"{wind_vals[i].item():.6f}"]
                    + [f"{mean_actions[i, j]:.6f}" for j in range(act_dim)]
                )
                for i in range(mean_actions.shape[0])
            ]
            sweep_path.write_text("\n".join([header] + body_lines))
            logger.info(f"Saved sweep table to {sweep_path}")


if __name__ == "__main__":
    main()
