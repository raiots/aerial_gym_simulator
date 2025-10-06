import torch


class task_config:
    seed = 56
    sim_name = "base_sim"
    env_name = "empty_env"
    robot_name = "trirotor"
    controller_name = "servo_thrust_split_trirotor"
    args = {}
    num_envs = 32
    use_warp = False
    headless = True
    device = "cuda:0"
    privileged_observation_space_dim = 0
    # 6 actions: [servo0, servo1, servo2, thrust0, thrust1, thrust2]
    action_space_dim = 6
    observation_space_dim = 15
    episode_len_steps = 600
    return_state_before_reset = False
    reward_parameters = {}
    crash_dist = 6.5
    # Debug: log first-agent motor thrusts every N steps (0 disables)
    log_first_agent_thrust_every_n_steps = 100

    # Per-dimension limits: first 3 are servo angles (rad), next 3 are thrusts (N)
    action_limit_max = torch.tensor([0.35, 0.35, 0.35, 45, 45, 45], device=device)
    action_limit_min = torch.tensor([-0.35, -0.35, -0.35, 0.0, 0.0, 0.0], device=device)

    def process_actions_for_task(actions, min_limit, max_limit):
        actions_clipped = torch.clamp(actions, -1.0, 1.0)
        # elementwise affine scaling to [min_limit, max_limit]
        return actions_clipped * (max_limit - min_limit) / 2.0 + (max_limit + min_limit) / 2.0
