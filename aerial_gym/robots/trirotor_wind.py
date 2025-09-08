import torch

from aerial_gym.robots.base_reconfigurable import BaseReconfigurable
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import quat_rotate, quat_rotate_inverse


logger = CustomLogger("trirotor_wind")


class TrirotorWithWind(BaseReconfigurable):
    """
    Trirotor subclass with a simple aerodynamic wind disturbance model.

    Assumptions:
    - Fixed wind direction in world frame (from +X direction).
    - Wind speed sampled uniformly in [wind_speed_min, wind_speed_max] per-episode.
    - Aerodynamic forces depend on pitch angle (small-angle model):
        C_L = cl_alpha * pitch,  C_D = cd0 + cd_alpha2 * pitch^2
      Forces applied in body frame: Drag along -X_b, Lift along +Z_b, then rotated to world.
    """

    def __init__(self, robot_config, controller_name, env_config, device):
        super().__init__(
            robot_config=robot_config,
            controller_name=controller_name,
            env_config=env_config,
            device=device,
        )
        # Aerodynamic params
        aero = self.cfg.aerodynamic_config
        # Always enable wind for this subclass
        self.wind_enabled = True
        self.rho = torch.tensor(aero.air_density, device=self.device)
        self.S = torch.tensor(aero.reference_area, device=self.device)
        self.cl_alpha = torch.tensor(aero.cl_alpha, device=self.device)
        self.cd0 = torch.tensor(aero.cd0, device=self.device)
        self.cd_alpha2 = torch.tensor(aero.cd_alpha2, device=self.device)
        self.wind_dir_world = torch.tensor(aero.wind_dir_world, device=self.device, dtype=torch.float32)
        self.wind_dir_world = self.wind_dir_world / torch.norm(self.wind_dir_world)
        self.wind_speed_min = aero.wind_speed_min
        self.wind_speed_max = aero.wind_speed_max

        # Will be initialized in init_tensors
        self.wind_speed = None
        self.wind_vec_world = None

    def init_tensors(self, global_tensor_dict):
        super().init_tensors(global_tensor_dict)
        # Sample wind speeds per env
        self.wind_speed = (
            torch.rand((self.num_envs,), device=self.device) * (self.wind_speed_max - self.wind_speed_min)
            + self.wind_speed_min
        )
        self.wind_vec_world = self.wind_dir_world.expand(self.num_envs, 3) * self.wind_speed.unsqueeze(-1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Resample wind for reset envs
        if self.wind_speed is not None:
            self.wind_speed[env_ids] = (
                torch.rand((len(env_ids),), device=self.device)
                * (self.wind_speed_max - self.wind_speed_min)
                + self.wind_speed_min
            )
            self.wind_vec_world[env_ids] = self.wind_dir_world.expand(len(env_ids), 3) * self.wind_speed[env_ids].unsqueeze(-1)

    def apply_wind_aerodynamics(self):
        if not self.wind_enabled:
            return

        # Relative flow in world frame (wind - robot linear velocity)
        v_rel_world = self.wind_vec_world - self.robot_linvel
        v_rel_mag = torch.clamp(torch.norm(v_rel_world, dim=-1), min=1e-6)
        q = 0.5 * self.rho * (v_rel_mag**2)  # dynamic pressure per env

        # Pitch angle from robot_euler_angles (ZYX -> [roll, pitch, yaw] ordering used upstream)
        pitch = self.robot_euler_angles[:, 1]
        C_L = self.cl_alpha * pitch
        C_D = self.cd0 + self.cd_alpha2 * pitch * pitch

        # Aerodynamic forces in body frame (Drag along -X_b, Lift along +Z_b)
        # F_body = [-C_D * q * S, 0, C_L * q * S]
        F_drag = -C_D * q * self.S
        F_lift = C_L * q * self.S
        F_body = torch.stack([F_drag, torch.zeros_like(F_drag), F_lift], dim=-1)

        # Rotate to world frame and apply at root link
        F_world = quat_rotate(self.robot_orientation, F_body)
        self.robot_force_tensors[:, 0, 0:3] += F_world

    def step(self, action_tensor):
        super().step(action_tensor)
        self.apply_wind_aerodynamics()
