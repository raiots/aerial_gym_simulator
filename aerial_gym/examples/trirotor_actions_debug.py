from aerial_gym.utils.logging import CustomLogger
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.helpers import get_args
import torch


logger = CustomLogger(__name__)


def read_dof_positions(env_manager):
    dof_states = env_manager.global_tensor_dict.get("dof_state_tensor", None)
    if dof_states is None or dof_states.numel() == 0:
        return None
    return dof_states[..., 0]


if __name__ == "__main__":
    args = get_args(
        additional_parameters=[
            {"name": "--tilt", "type": float, "default": 0.2, "help": "servo tilt angle [rad]"},
            {"name": "--thrust", "type": float, "default": 0.35, "help": "motor thrust [N]"},
        ]
    )

    logger.info("Building trirotor actions debug environment (no gravity).")
    env_manager = SimBuilder().build_env(
        sim_name="base_sim_no_gravity",
        env_name="empty_env",
        robot_name="trirotor_wind",
        controller_name="servo_thrust_split_trirotor",
        args=None,
        device=args.sim_device,
        num_envs=args.num_envs,
        headless=args.headless,
        use_warp=args.use_warp,
    )

    # Actions: [servo0, servo1, servo2, motor0, motor1, motor2]
    actions = torch.zeros((env_manager.num_envs, 6), device=args.sim_device)
    env_manager.reset()

    # Phase 1: zero thrust, zero tilt
    logger.info("Phase 1: zero thrust, zero tilt.")
    actions[:] = 0.0
    for _ in range(200):
        env_manager.step(actions=actions)

    # Phase 2: set uniform tilt, thrust remains zero
    logger.info("Phase 2: zero thrust, uniform tilt.")
    actions[:, 0:3] = args.tilt
    actions[:, 3:6] = 0.0
    for i in range(200):
        env_manager.step(actions=actions)
        if i % 50 == 0:
            dof = read_dof_positions(env_manager)
            if dof is not None:
                logger.info(f"tilt targets ~ dof pos: {dof[0, :3].tolist()}")

    # Phase 3: small thrust + uniform tilt
    logger.info("Phase 3: small thrust, uniform tilt -> observe motion.")
    actions[:, 3:6] = args.thrust
    for i in range(4000):
        env_manager.step(actions=actions)
        if i % 50 == 0:
            pos = env_manager.global_tensor_dict["robot_position"][0].tolist()
            dof = read_dof_positions(env_manager)
            logger.info(f"t={i} pos={pos} dof={None if dof is None else dof[0, :3].tolist()}")
