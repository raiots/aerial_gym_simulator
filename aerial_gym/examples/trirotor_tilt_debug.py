from aerial_gym.utils.logging import CustomLogger
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.helpers import get_args
import torch


logger = CustomLogger(__name__)


def set_servo_targets(env_manager, angle_rad):
    dof_targets = env_manager.global_tensor_dict["dof_position_setpoint_tensor"]
    if dof_targets.numel() == 0:
        raise RuntimeError("Robot has no DOFs; expected 3 tilt joints.")
    dof_targets[:] = angle_rad


if __name__ == "__main__":
    args = get_args(
        additional_parameters=[
            {"name": "--tilt", "type": float, "default": 0.2, "help": "servo tilt angle [rad]"},
        ]
    )

    logger.info("Building trirotor debug environment (no gravity).")
    env_manager = SimBuilder().build_env(
        sim_name="base_sim_no_gravity",
        env_name="empty_env",
        robot_name="trirotor",
        controller_name="no_control_trirotor",
        args=None,
        device=args.sim_device,
        num_envs=args.num_envs,
        headless=args.headless,
        use_warp=args.use_warp,
    )

    actions = torch.zeros((env_manager.num_envs, 3), device=args.sim_device)
    env_manager.reset()

    logger.info("Phase 1: zero thrust, zero tilt.")
    set_servo_targets(env_manager, 0.0)
    for _ in range(400):
        env_manager.step(actions=actions)

    logger.info("Phase 2: zero thrust, apply uniform tilt.")
    set_servo_targets(env_manager, args.tilt)
    for _ in range(1800):
        env_manager.step(actions=actions)

    logger.info("Phase 3: small thrust, uniform tilt -> observe motion.")
    actions[:] = 0.35  # small thrust on all 3 motors
    for i in range(800):
        env_manager.step(actions=actions)
        if i % 50 == 0:
            pos = env_manager.global_tensor_dict["robot_position"][0].tolist()
            logger.info(f"t={i} pos={pos}")
