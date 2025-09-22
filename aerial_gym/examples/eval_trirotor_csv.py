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
        # Detect motor thrust tensor availability to size cmd/actual columns
        mt = env.obs_dict.get("motor_thrusts", None)
        motor_dim = int(mt.shape[1]) if (mt is not None and mt.ndim == 2) else 0

        header = [
            "episode","ep_step","done","reward","t",
            "pos_x","pos_y","pos_z",
            "roll","pitch","yaw",
            "vel_x","vel_y","vel_z",
            "ang_x","ang_y","ang_z",
        ]
        # Raw policy outputs (mu) for next step
        header += [f"act_{i}" for i in range(act_dim)]
        # Also log clipped and scaled versions derived from mu
        header += [f"act_clipped_{i}" for i in range(act_dim)]
        header += [f"act_scaled_{i}" for i in range(act_dim)]
        # If it's a 6D action space, split into servo (rad/deg) and motor (N)
        if act_dim >= 6:
            header += [f"servo_next_{i}_rad" for i in range(3)]
            header += [f"servo_next_{i}_deg" for i in range(3)]
            header += [f"motor_next_{i}_N" for i in range(3)]
            # Also log last applied servo targets from sim, if available
            header += [f"servo_cmd_last_{i}_rad" for i in range(3)]
        else:
            # 3D action space → all actions are motors
            header += [f"motor_next_{i}_N" for i in range(act_dim)]
        # Commanded and actual motor thrusts from the simulator (last step)
        if motor_dim > 0:
            header += [f"motor_cmd_last_{i}_N" for i in range(motor_dim)]
            header += [f"motor_actual_last_{i}_N" for i in range(motor_dim)]
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
                # For action post-processing (clipped/scaled) we reuse task_config mapping
                # Note: scaled corresponds to physical units (servo [rad], motor [N])
                act_mu = actions.detach()
                act_clipped = torch.clamp(act_mu, -1.0, 1.0)
                # Ensure device alignment with task limits
                lim_min = env.task_config.action_limit_min
                lim_max = env.task_config.action_limit_max
                act_scaled = env.task_config.process_actions_for_task(
                    act_clipped.to(lim_min.device), lim_min, lim_max
                )

                # Split into servo and motor for 6D; else all are motors
                servo_next_rad = None
                servo_next_deg = None
                motor_next_N = None
                if act_dim >= 6:
                    servo_next_rad = act_scaled[..., 0:3].detach().cpu()
                    servo_next_deg = servo_next_rad * 180.0 / np.pi
                    motor_next_N = act_scaled[..., 3:6].detach().cpu()
                else:
                    motor_next_N = act_scaled.detach().cpu()

                # DoF servo command actually applied in last step (if available)
                dof_cmd = None
                try:
                    dof_cmd_tensor = env.sim_env.global_tensor_dict.get(
                        "dof_position_setpoint_tensor", None
                    )
                    if dof_cmd_tensor is not None and dof_cmd_tensor.numel() > 0:
                        dof_cmd = dof_cmd_tensor.detach().cpu()
                except Exception:
                    dof_cmd = None

                # Commanded/actual motor thrusts from simulator (last step)
                mt_cmd = od.get("motor_thrusts_cmd", None)
                mt_act = od.get("motor_thrusts", None)
                if mt_cmd is not None:
                    mt_cmd = mt_cmd.detach().cpu()
                if mt_act is not None:
                    mt_act = mt_act.detach().cpu()

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
                ]
                # Raw mu for next step (kept as legacy act_i columns)
                row += [f"{act_mu[idx,i].item():.6f}" for i in range(act_dim)]
                # Clipped and scaled versions
                row += [f"{act_clipped[idx,i].item():.6f}" for i in range(act_dim)]
                row += [f"{act_scaled[idx,i].item():.6f}" for i in range(act_dim)]
                # Servo/motor split
                if act_dim >= 6:
                    row += [f"{servo_next_rad[idx,i].item():.6f}" for i in range(3)]
                    row += [f"{servo_next_deg[idx,i].item():.6f}" for i in range(3)]
                    row += [f"{motor_next_N[idx,i].item():.6f}" for i in range(3)]
                    if dof_cmd is not None:
                        row += [f"{dof_cmd[idx,i].item():.6f}" for i in range(3)]
                    else:
                        row += ["" for _ in range(3)]
                else:
                    row += [f"{motor_next_N[idx,i].item():.6f}" for i in range(act_dim)]

                # Motor thrusts commanded/actual from simulator (last step)
                if motor_dim > 0:
                    if mt_cmd is not None:
                        row += [f"{mt_cmd[idx,i].item():.6f}" for i in range(motor_dim)]
                    else:
                        row += ["" for _ in range(motor_dim)]
                    if mt_act is not None:
                        row += [f"{mt_act[idx,i].item():.6f}" for i in range(motor_dim)]
                    else:
                        row += ["" for _ in range(motor_dim)]
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
