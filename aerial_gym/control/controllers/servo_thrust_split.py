from aerial_gym.utils.logging import CustomLogger
import torch


logger = CustomLogger("servo_thrust_split_controller")


class ServoThrustSplitController:
    """
    Controller that treats the first N_dof action components as DOF position setpoints
    (servo angles) and the remaining components as motor thrust commands.

    Intended for trirotor debugging where N_dof=3 and N_motors=3.
    """

    def __init__(self, config, num_envs, device):
        self.cfg = config
        self.num_envs = num_envs
        self.device = device
        self.num_actions = self.cfg.num_actions  # e.g., 6 (3 servos + 3 motors)
        # Controller outputs per-motor thrusts to be applied directly
        self.output_mode = "forces"

        # Derived sizes for split (default: equal split into 3 + 3)
        # For flexibility, allow overriding via config attributes if present
        self.num_dof_actions = getattr(self.cfg, "num_dof_actions", self.num_actions // 2)
        self.num_motor_actions = self.num_actions - self.num_dof_actions

        self.dof_position_setpoint_tensor = None
        self.dof_velocity_setpoint_tensor = None

    def init_tensors(self, global_tensor_dict=None):
        # Store ref; BaseReconfigurable adds DOF tensors after controller.init_tensors runs
        self.global_tensor_dict = global_tensor_dict
        self.dof_position_setpoint_tensor = global_tensor_dict.get(
            "dof_position_setpoint_tensor", None
        )
        self.dof_velocity_setpoint_tensor = global_tensor_dict.get(
            "dof_velocity_setpoint_tensor", None
        )
        return

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def reset_commands(self):
        pass

    def reset(self):
        self.reset_idx(env_ids=None)

    def reset_idx(self, env_ids):
        pass

    def randomize_params(self, env_ids):
        pass

    def update(self, command_actions: torch.Tensor) -> torch.Tensor:
        # Expect [servo_0..servo_{k-1}, motor_0..motor_{m-1}]
        if command_actions.shape[1] != self.num_actions:
            raise ValueError(
                f"Expected action dim {self.num_actions}, got {command_actions.shape[1]}"
            )
        servo_targets = command_actions[:, : self.num_dof_actions]
        motor_thrusts = command_actions[:, self.num_dof_actions :]

        # Lazy fetch DOF tensors if not yet bound
        if self.dof_position_setpoint_tensor is None and self.global_tensor_dict is not None:
            self.dof_position_setpoint_tensor = self.global_tensor_dict.get(
                "dof_position_setpoint_tensor", None
            )
            self.dof_velocity_setpoint_tensor = self.global_tensor_dict.get(
                "dof_velocity_setpoint_tensor", None
            )

        if self.dof_position_setpoint_tensor is not None:
            # Broadcast into DOF tensor if sizes match
            if self.dof_position_setpoint_tensor.shape[1] >= self.num_dof_actions:
                self.dof_position_setpoint_tensor[:, : self.num_dof_actions] = servo_targets
            else:
                logger.error(
                    f"DOF tensor smaller than expected: {self.dof_position_setpoint_tensor.shape}"
                )
        if self.dof_velocity_setpoint_tensor is not None:
            self.dof_velocity_setpoint_tensor[:, : self.num_dof_actions] = 0.0

        # Return motor thrusts to be allocated by the robot
        return motor_thrusts
