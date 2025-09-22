import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from importlib.machinery import SourceFileLoader

# Lazy-load MLP loader directly from file to avoid importing aerial_gym package (__init__ imports isaacgym)
_mlp_loader = SourceFileLoader(
    "rl_games_inference",
    str(Path(__file__).resolve().parents[2] / "examples" / "rl_games_example" / "rl_games_inference.py"),
)
rl_inf = _mlp_loader.load_module()
MLP = rl_inf.MLP

# Only import task registry when needed for online mode, to avoid isaacgym import in csv mode
import logging
logger = logging.getLogger("analyze_controller")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter('[%(relativeCreated)d ms][%(name)s] - %(levelname)s : %(message)s')
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def register_activation_hooks(model: nn.Module) -> Dict[str, List[torch.Tensor]]:
    """Register forward hooks on Linear/Activation layers to capture activations."""
    activations: Dict[str, List[torch.Tensor]] = {}

    def hook_fn(name):
        def fn(_module, _inp, out):
            if isinstance(out, torch.Tensor):
                activations.setdefault(name, []).append(out.detach().cpu())
        return fn

    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.ELU, nn.ReLU, nn.Tanh, nn.Sigmoid)):
            m.register_forward_hook(hook_fn(name))
    return activations


def compute_jacobian(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Compute jacobian d(model(x))/dx at a single input x. Returns [out_dim, in_dim]."""
    model.eval()
    x = x.detach().clone().requires_grad_(True)
    y = model(x)
    if y.ndim == 2 and y.shape[0] == 1:
        y = y[0]
    out_dim = y.shape[0]
    in_dim = x.shape[-1]
    J = torch.zeros((out_dim, in_dim), device=x.device)
    for i in range(out_dim):
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        y[i].backward(retain_graph=True)
        J[i, :] = x.grad.detach().clone()
    return J.detach()


def save_jacobian(J: torch.Tensor, obs_labels: List[str], act_labels: List[str], out_path: Path):
    df = pd.DataFrame(J.cpu().numpy(), index=act_labels, columns=obs_labels)
    df.to_csv(out_path, index=True)


def plot_jacobian(
    J: torch.Tensor,
    out_path: Path,
    title: str = "Jacobian",
    obs_labels: Optional[List[str]] = None,
    act_labels: Optional[List[str]] = None,
):
    data = J.cpu().numpy()
    n_act, n_obs = data.shape
    plt.figure(figsize=(max(8, n_obs * 0.5), max(4, n_act * 0.6)))
    im = plt.imshow(data, aspect='auto', cmap='coolwarm')
    cbar = plt.colorbar(im, label='∂a/∂x')
    plt.title(title)
    if obs_labels is not None and len(obs_labels) == n_obs:
        plt.xticks(ticks=np.arange(n_obs), labels=obs_labels, rotation=45, ha='right', fontsize=8)
        plt.xlabel('Observation')
    else:
        plt.xlabel('Observation index')
    if act_labels is not None and len(act_labels) == n_act:
        plt.yticks(ticks=np.arange(n_act), labels=act_labels, fontsize=9)
        plt.ylabel('Action')
    else:
        plt.ylabel('Action index')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_activations(activations: Dict[str, List[torch.Tensor]], out_dir: Path):
    for name, seq in activations.items():
        if len(seq) == 0:
            continue
        A = torch.stack([t.flatten() for t in seq], dim=0).numpy()
        df = pd.DataFrame(A)
        df.to_csv(out_dir / f"activations__{name.replace('.', '_')}.csv", index=False)


def plot_activation_histograms(activations: Dict[str, List[torch.Tensor]], out_dir: Path, bins: int = 50):
    for name, seq in activations.items():
        if len(seq) == 0:
            continue
        vals = torch.cat([t.flatten() for t in seq], dim=0).numpy()
        plt.figure(figsize=(6, 4))
        plt.hist(vals, bins=bins, color='tab:blue', alpha=0.8)
        plt.title(f"Activation histogram — {name}")
        plt.xlabel("Activation value")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / f"act_hist__{name.replace('.', '_')}.pdf", dpi=300)
        plt.close()


def get_act_obs_labels(act_dim: int, obs_dim: int, include_wind: bool) -> Tuple[List[str], List[str]]:
    # Observations: [pos_err(3), rot6d(6), linvel(3), angvel(3)] + wind(optional)
    obs_labels = [
        "pos_ex", "pos_ey", "pos_ez",
        "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
        "vel_x", "vel_y", "vel_z",
        "ang_x", "ang_y", "ang_z",
    ]
    if include_wind:
        obs_labels.append("wind_speed")
    # Trim/extend to obs_dim just in case
    if len(obs_labels) < obs_dim:
        obs_labels += [f"obs_{i}" for i in range(len(obs_labels), obs_dim)]
    else:
        obs_labels = obs_labels[:obs_dim]

    # Actions: if 6D -> [servo0, servo1, servo2, motor0, motor1, motor2], else motor0.. motor_{act_dim-1}
    if act_dim >= 6:
        act_labels = ["servo_0", "servo_1", "servo_2", "motor_0", "motor_1", "motor_2"]
        act_labels = act_labels[:act_dim]
    else:
        act_labels = [f"motor_{i}" for i in range(act_dim)]
    return act_labels, obs_labels


def analyze_online(args):
    results_dir = Path(args.out)
    ensure_dir(results_dir)

    # Build env
    from aerial_gym.registry.task_registry import task_registry
    env = task_registry.make_task(args.task, num_envs=args.num_envs, headless=True, use_warp=False)
    env.reset()

    obs_dim = env.task_config.observation_space_dim
    act_dim = env.task_config.action_space_dim
    include_wind = ("wind_speed" in env.obs_dict)
    act_labels, obs_labels = get_act_obs_labels(act_dim, obs_dim, include_wind)

    device = torch.device(args.device)
    # Load policy
    model = MLP(obs_dim, act_dim, args.checkpoint).to(device).eval()
    activations = register_activation_hooks(model)

    actions = torch.zeros((env.sim_env.num_envs, act_dim), device=device)
    # We will store Jacobians for a subset of steps
    # Choose steps for jacobian computation: --indices or evenly spaced
    if args.indices:
        jacobian_steps = parse_index_spec(args.indices, args.steps)
        logger.info(f"Online mode: using user-specified steps for Jacobian: {jacobian_steps}")
    else:
        jacobian_steps = np.linspace(0, args.steps - 1, num=min(args.jacobians, args.steps), dtype=int).tolist()
        logger.info(f"Online mode: using evenly spaced {len(jacobian_steps)} steps for Jacobian: {jacobian_steps}")
    jac_ctr = 0

    with torch.no_grad():
        for step in range(args.steps):
            # step env using previous actions
            obs, rew, term, trunc, info = env.step(actions=actions)
            x = obs["observations"][0:1].to(device)
            # forward pass (hooks capture activations)
            y = model(x)
            actions[:] = y

            # Compute jacobian on selected steps (with grad)
            if step in jacobian_steps:
                # Temporarily enable grad for jacobian
                for p in model.parameters():
                    p.requires_grad_(False)
                J = compute_jacobian(model, x)
                save_jacobian(J, obs_labels, act_labels, results_dir / f"jacobian_step_{step:05d}.csv")
                plot_jacobian(J, results_dir / f"jacobian_step_{step:05d}.pdf", title=f"Jacobian @ step {step}", obs_labels=obs_labels, act_labels=act_labels)
                jac_ctr += 1

    # Save activations
    save_activations(activations, results_dir)
    plot_activation_histograms(activations, results_dir)
    logger.info(f"Saved analysis to {results_dir}")


def analyze_from_csv(args):
    results_dir = Path(args.out)
    ensure_dir(results_dir)

    df = pd.read_csv(args.csv)
    # Reconstruct observation matrix in the same order as used by the policy
    # Here we expect columns: pos_x,pos_y,pos_z, roll,pitch,yaw, vel*, ang*, plus actions/wind if present
    # But eval_trirotor_csv stored processed observations implicitly; we will approximate using available columns.
    # Better approach: capture obs directly during eval logging — can be added later.

    # Fallback: build obs from available columns (pos error unavailable offline). Users can plug in their saved obs.
    logger.warning("CSV mode: building observations from state columns; results may differ from training obs pipeline.")

    cols = []
    for c in ["pos_x", "pos_y", "pos_z"]:
        if c in df.columns:
            cols.append(c)
    # Placeholder for rot6d (not directly available) — skip or zeros
    rot6d = np.zeros((len(df), 6), dtype=np.float32)
    vel_cols = [c for c in ["vel_x", "vel_y", "vel_z"] if c in df.columns]
    ang_cols = [c for c in ["ang_x", "ang_y", "ang_z"] if c in df.columns]

    obs_parts = []
    if cols:
        obs_parts.append(df[cols].values.astype(np.float32))
    obs_parts.append(rot6d)
    if len(vel_cols) == 3:
        obs_parts.append(df[vel_cols].values.astype(np.float32))
    else:
        obs_parts.append(np.zeros((len(df), 3), dtype=np.float32))
    if len(ang_cols) == 3:
        obs_parts.append(df[ang_cols].values.astype(np.float32))
    else:
        obs_parts.append(np.zeros((len(df), 3), dtype=np.float32))

    if "wind_speed" in df.columns:
        obs_parts.append(df[["wind_speed"]].values.astype(np.float32))

    obs_mat = np.concatenate(obs_parts, axis=1)

    device = torch.device(args.device)
    # Try to infer dimensions from checkpoint by user-provided obs_dim if needed
    if args.obs_dim is None:
        obs_dim = obs_mat.shape[1]
    else:
        obs_dim = args.obs_dim
        if obs_mat.shape[1] < obs_dim:
            pad = np.zeros((obs_mat.shape[0], obs_dim - obs_mat.shape[1]), dtype=np.float32)
            obs_mat = np.concatenate([obs_mat, pad], axis=1)
        elif obs_mat.shape[1] > obs_dim:
            obs_mat = obs_mat[:, :obs_dim]

    act_dim = args.act_dim if args.act_dim is not None else 3
    model = MLP(obs_dim, act_dim, args.checkpoint).to(device).eval()
    activations = register_activation_hooks(model)

    # Select sample points: --indices or evenly spaced
    if args.indices:
        idxs = parse_index_spec(args.indices, len(obs_mat))
        logger.info(f"CSV mode: using user-specified indices for Jacobian: {idxs}")
    else:
        idxs = np.linspace(0, len(obs_mat) - 1, num=min(args.jacobians, len(obs_mat)), dtype=int).tolist()
        logger.info(f"CSV mode: using evenly spaced {len(idxs)} indices for Jacobian: {idxs}")
    for s, idx in enumerate(idxs):
        x = torch.from_numpy(obs_mat[idx:idx + 1]).to(device)
        with torch.no_grad():
            _ = model(x)
        J = compute_jacobian(model, x)
        act_labels, obs_labels = get_act_obs_labels(act_dim, obs_dim, include_wind=("wind_speed" in df.columns))
        save_jacobian(J, obs_labels, act_labels, results_dir / f"jacobian_csv_{idx:05d}.csv")
        plot_jacobian(J, results_dir / f"jacobian_csv_{idx:05d}.pdf", title=f"Jacobian @ csv idx {idx}", obs_labels=obs_labels, act_labels=act_labels)

    save_activations(activations, results_dir)
    plot_activation_histograms(activations, results_dir)
    logger.info(f"Saved analysis to {results_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="NN Controller Explainability: local linearization & activations")
    p.add_argument("--checkpoint", type=str, required=True, help="RL-Games checkpoint .pth")
    p.add_argument("--task", type=str, default="position_setpoint_task_sim2real_end_to_end")
    p.add_argument("--mode", type=str, choices=["online", "csv"], default="online")
    p.add_argument("--csv", type=str, help="CSV file for offline mode")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--jacobians", type=int, default=10, help="number of jacobians to compute (ignored if --indices provided)")
    p.add_argument("--indices", type=str, default=None, help="explicit indices or steps to compute Jacobian, e.g. '10,50,120' or '100:1000:50'")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out", type=str, default="results/nn_explainability")
    # For CSV mode dimension hints
    p.add_argument("--obs_dim", type=int, default=None)
    p.add_argument("--act_dim", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.mode == "online":
        analyze_online(args)
    else:
        if not args.csv:
            raise ValueError("--csv is required for mode=csv")
        analyze_from_csv(args)


def parse_index_spec(spec: str, length: int) -> List[int]:
    """Parse index specification like '10,50,120' or '100:1000:50'. Clamp to [0, length-1]."""
    spec = spec.strip()
    out: List[int] = []
    if ":" in spec and "," not in spec:
        # slice-style: start:end[:step]
        parts = spec.split(":")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            step = 1
        elif len(parts) == 3:
            start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            raise ValueError(f"Invalid --indices slice spec: {spec}")
        if step == 0:
            raise ValueError("--indices step cannot be 0")
        rng = range(start, end + (1 if step > 0 else -1), step)
        out = list(rng)
    else:
        # comma or space separated list
        tokens = [t for t in spec.replace(" ", ",").split(",") if t]
        out = [int(t) for t in tokens]
    # clamp and unique-sort
    out = [max(0, min(length - 1, i)) for i in out]
    out = sorted(list(dict.fromkeys(out)))
    return out


if __name__ == "__main__":
    main()
