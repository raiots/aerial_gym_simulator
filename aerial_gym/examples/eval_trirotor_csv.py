import argparse
import time
import csv
import numpy as np

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.examples.rl_games_example.rl_games_inference import MLP
import torch
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_euler_angles


logger = CustomLogger("eval_trirotor_csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="position_setpoint_task_sim2real_end_to_end",
                   help="Task name registered in task_registry (e.g., position_setpoint_task_sim2real_end_to_end or position_setpoint_task_trirotor_6d)")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to RL-Games checkpoint .pth")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--headless", type=lambda x: str(x).lower() in ["1","true","yes"], default=True)
    p.add_argument("--use_warp", type=lambda x: str(x).lower() in ["1","true","yes"], default=False)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--csv", type=str, default="trirotor_eval_log.csv")
    return p.parse_args()


def quat_xyz_to_rpy_zxy(quat_tensor):
    # input shape: [N, 4] where stored as [x,y,z,w]; convert to [w,x,y,z]
    q = quat_tensor[:, [3, 0, 1, 2]]
    R = quaternion_to_matrix(q)
    euler = matrix_to_euler_angles(R, "ZYX")  # returns [yaw, pitch, roll]
    # reorder to roll, pitch, yaw
    rpy = torch.stack([euler[:, 2], euler[:, 1], euler[:, 0]], dim=-1)
    return rpy


def main():
    args = parse_args()
    torch.cuda.empty_cache()

    # Build task
    logger.info(f"Creating task={args.task}, num_envs={args.num_envs}, headless={args.headless}, use_warp={args.use_warp}")
    env = task_registry.make_task(args.task, num_envs=args.num_envs, headless=args.headless, use_warp=args.use_warp)
    env.reset()

    obs_dim = env.task_config.observation_space_dim
    act_dim = env.task_config.action_space_dim
    device = torch.device(args.device)

    # Load policy (actor MLP) from RL-Games checkpoint
    model = MLP(obs_dim, act_dim, args.checkpoint).to(device).eval()

    actions = torch.zeros((env.sim_env.num_envs, act_dim), device=device)
    logger.info(f"Starting evaluation for {args.steps} steps; logging to {args.csv}")

    # Prepare CSV
    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "episode","ep_step","done","reward","t",
            "pos_x","pos_y","pos_z",
            "roll","pitch","yaw",
            "vel_x","vel_y","vel_z",
            "ang_x","ang_y","ang_z",
        ] + [f"act_{i}" for i in range(act_dim)]
        writer.writerow(header)

        t0 = time.time()
        # Episode bookkeeping per env
        ep_id = torch.zeros(env.sim_env.num_envs, dtype=torch.long)
        ep_step = torch.zeros(env.sim_env.num_envs, dtype=torch.long)
        with torch.no_grad():
            for step in range(args.steps):
                # Step env using previous actions (start with zeros)
                obs, rew, term, trunc, info = env.step(actions=actions)
                # Compute new actions from observations
                obs_tensor = obs["observations"].to(device)
                actions[:] = model.forward(obs_tensor)

                # Extract state (env.obs_dict holds tensors from sim)
                od = env.obs_dict
                pos = od["robot_position"][..., :3].detach().cpu()
                quat = od["robot_orientation"][..., :4].detach().cpu()
                rpy = quat_xyz_to_rpy_zxy(quat)
                lin = od["robot_linvel"][..., :3].detach().cpu()
                ang = od["robot_body_angvel"][..., :3].detach().cpu()

                # Log only env 0 for clarity; extend as needed
                idx = 0
                now = time.time() - t0
                done_flags = (term | trunc).detach().cpu()
                row = [
                    int(ep_id[idx].item()), int(ep_step[idx].item()), int(done_flags[idx].item()), f"{rew[idx].item():.6f}", f"{now:.4f}",
                    f"{pos[idx,0].item():.6f}", f"{pos[idx,1].item():.6f}", f"{pos[idx,2].item():.6f}",
                    f"{rpy[idx,0].item():.6f}", f"{rpy[idx,1].item():.6f}", f"{rpy[idx,2].item():.6f}",
                    f"{lin[idx,0].item():.6f}", f"{lin[idx,1].item():.6f}", f"{lin[idx,2].item():.6f}",
                    f"{ang[idx,0].item():.6f}", f"{ang[idx,1].item():.6f}", f"{ang[idx,2].item():.6f}",
                ] + [f"{actions[idx,i].item():.6f}" for i in range(act_dim)]
                writer.writerow(row)

                # Update episode counters after logging this row
                for i in range(env.sim_env.num_envs):
                    if int(done_flags[i].item()) == 1:
                        ep_id[i] += 1
                        ep_step[i] = 0
                    else:
                        ep_step[i] += 1

    logger.info(f"CSV saved: {args.csv}")


if __name__ == "__main__":
    main()
