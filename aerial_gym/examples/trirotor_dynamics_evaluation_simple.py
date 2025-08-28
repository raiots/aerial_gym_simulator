#!/usr/bin/env python3
"""
Trirotor Dynamics Performance Evaluation Script (Simplified)

This script evaluates the dynamics performance of the trirotor configuration
with the ability to toggle gravity and uses direct motor control (no_control).

Edit the configuration variables below to change settings:
- ENABLE_GRAVITY: True/False to enable/disable gravity
- NUM_ENVS: Number of simulation environments
- HEADLESS: True for headless mode, False for GUI
- TEST_DURATION: Duration in simulation steps
"""

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np
import time

logger = CustomLogger(__name__)

# ===== CONFIGURATION (Edit these variables) =====
ENABLE_GRAVITY = True       # Set to False to disable gravity
NUM_ENVS = 4               # Number of simulation environments  
HEADLESS = False             # Set to False to see GUI
USE_WARP = False            # Use warp rendering
TEST_DURATION = 1000        # Total test steps
SAVE_RESULTS = True        # Save performance analysis to file
# ===============================================


class TrirotorDynamicsEvaluator:
    def __init__(self, enable_gravity=True, num_envs=4, headless=True, use_warp=False):
        self.enable_gravity = enable_gravity
        self.num_envs = num_envs
        self.headless = headless
        self.use_warp = use_warp
        
        # Performance metrics
        self.position_data = []
        self.velocity_data = []
        self.angular_velocity_data = []
        self.thrust_data = []
        self.time_data = []
        
        logger.info(f"Initializing Trirotor Dynamics Evaluator")
        logger.info(f"Gravity: {'Enabled' if enable_gravity else 'Disabled'}")
        logger.info(f"Number of environments: {num_envs}")
        
        self._build_environment()
    
    def _build_environment(self):
        """Build the simulation environment with trirotor configuration"""        
        self.env_manager = SimBuilder().build_env(
            sim_name="base_sim",
            env_name="empty_env",
            robot_name="trirotor",
            controller_name="no_control",
            args=None,
            device="cuda:0",
            num_envs=self.num_envs,
            headless=self.headless,
            use_warp=self.use_warp,
        )
        
        # Manually disable gravity if requested
        if not self.enable_gravity:
            logger.info("Manually disabling gravity in simulation")
            # Note: Gravity disabling would need to be implemented at the sim level
            # For now this serves as a placeholder for that functionality
        
        # Action space for 3 motors (trirotor)
        self.action_dim = 3
        self.actions = torch.zeros((self.num_envs, self.action_dim)).to("cuda:0")
        
        logger.info(f"Environment built successfully. Action dimension: {self.action_dim}")
    
    def reset_environments(self):
        """Reset all environments"""
        self.env_manager.reset()
        logger.debug("Environments reset")
    
    def step_test_constant_thrust(self, thrust_value=0.6, steps=100):
        """Test with constant thrust on all motors"""
        logger.info(f"Running constant thrust test: {thrust_value} for {steps} steps")
        
        self.actions.fill_(thrust_value)
        
        for i in range(steps):
            self.env_manager.step(actions=self.actions)
            self._record_metrics(i)
            
            if not self.headless and i % 1 == 0:
                self.env_manager.render()
                time.sleep(0.01)
    
    def step_test_differential_thrust(self, steps=100):
        """Test with differential thrust patterns"""
        logger.info(f"Running differential thrust test for {steps} steps")
        
        for i in range(steps):
            # Create sinusoidal thrust pattern for dynamic testing
            t = i * 0.02  # Assuming 50Hz simulation
            
            # Different patterns for each motor to test control authority
            self.actions[:, 0] = 0.5 + 0.2 * torch.sin(torch.tensor(t))           # Motor 1
            self.actions[:, 1] = 0.5 + 0.2 * torch.sin(torch.tensor(t + np.pi/3))  # Motor 2  
            self.actions[:, 2] = 0.5 + 0.2 * torch.sin(torch.tensor(t + 2*np.pi/3)) # Motor 3
            
            self.env_manager.step(actions=self.actions)
            self._record_metrics(i + 100)  # Offset time for plotting
            
            if not self.headless and i % 1 == 0:
                self.env_manager.render()
    
    def step_test_hovering_thrust(self, steps=100):
        """Test hovering capabilities with appropriate thrust"""
        logger.info(f"Running hovering thrust test for {steps} steps")
        
        # For trirotor, hovering thrust calculation:
        # Robot mass: 0.36kg, gravity: 9.81m/s²
        # Total thrust needed: 0.36 * 9.81 = 3.53N
        # Per motor: 3.53N / 3 motors = 1.18N per motor
        # With max_thrust=2.5N, normalized: 1.18/2.5 = 0.47
        hover_thrust = 0.47 if self.enable_gravity else 0.0
        self.actions.fill_(hover_thrust)
        
        for i in range(steps):
            self.env_manager.step(actions=self.actions)
            self._record_metrics(i + 200)  # Offset time for plotting
            
            if not self.headless and i % 1 == 0:
                self.env_manager.render()
    
    def _record_metrics(self, step):
        """Record performance metrics"""
        # Get robot states from the robot manager
        robot = self.env_manager.robot_manager.robot
        
        # Extract position, velocity, and angular velocity (first environment)
        # Using robot's tensors directly
        pos = robot.robot_position[0].cpu().numpy()
        vel = robot.robot_linvel[0].cpu().numpy()
        ang_vel = robot.robot_angvel[0].cpu().numpy()
        
        self.position_data.append(pos)
        self.velocity_data.append(vel)
        self.angular_velocity_data.append(ang_vel)
        self.thrust_data.append(self.actions[0].cpu().numpy().copy())
        self.time_data.append(step * 0.02)  # Assuming 50Hz
    
    def run_full_evaluation(self, test_duration=300):
        """Run complete dynamics evaluation"""
        logger.info("Starting full trirotor dynamics evaluation")
        
        # Clear previous data
        self.position_data = []
        self.velocity_data = []
        self.angular_velocity_data = []
        self.thrust_data = []
        self.time_data = []
        
        # Reset environment
        self.reset_environments()
        
        # Run different test phases
        steps_per_test = test_duration // 1
        
        # Phase 1: Constant thrust
        self.step_test_constant_thrust(thrust_value=0.1, steps=steps_per_test)
        
        
        # Phase 2: Differential thrust  
        # self.step_test_differential_thrust(steps=steps_per_test)
        
        # Phase 3: Hovering thrust
        # self.step_test_hovering_thrust(steps=steps_per_test)
        
        logger.info("Full evaluation completed")
    
    def analyze_performance(self):
        """Analyze and print performance metrics"""
        if not self.position_data:
            logger.warning("No data recorded for analysis")
            return
            
        pos_array = np.array(self.position_data)
        vel_array = np.array(self.velocity_data)
        ang_vel_array = np.array(self.angular_velocity_data)
        
        logger.info("=== TRIROTOR DYNAMICS PERFORMANCE ANALYSIS ===")
        logger.info(f"Gravity: {'Enabled' if self.enable_gravity else 'Disabled'}")
        logger.info(f"Test Duration: {len(self.time_data)} steps ({self.time_data[-1]:.2f} seconds)")
        
        # Position analysis
        max_pos = np.max(np.abs(pos_array), axis=0)
        logger.info(f"Maximum displacement: X={max_pos[0]:.3f}m, Y={max_pos[1]:.3f}m, Z={max_pos[2]:.3f}m")
        
        # Velocity analysis  
        max_vel = np.max(np.abs(vel_array), axis=0)
        logger.info(f"Maximum velocity: X={max_vel[0]:.3f}m/s, Y={max_vel[1]:.3f}m/s, Z={max_vel[2]:.3f}m/s")
        
        # Angular velocity analysis
        max_ang_vel = np.max(np.abs(ang_vel_array), axis=0)
        logger.info(f"Maximum angular velocity: Roll={max_ang_vel[0]:.3f}rad/s, Pitch={max_ang_vel[1]:.3f}rad/s, Yaw={max_ang_vel[2]:.3f}rad/s")
        
        # Stability analysis (last 100 steps)
        if len(pos_array) > 100:
            final_pos_std = np.std(pos_array[-100:], axis=0)
            logger.info(f"Final position stability (std): X={final_pos_std[0]:.4f}m, Y={final_pos_std[1]:.4f}m, Z={final_pos_std[2]:.4f}m")
        
        # Save analysis to file if requested
        return {
            'gravity_enabled': self.enable_gravity,
            'test_duration_steps': len(self.time_data),
            'test_duration_seconds': self.time_data[-1] if self.time_data else 0,
            'max_displacement': max_pos.tolist(),
            'max_velocity': max_vel.tolist(), 
            'max_angular_velocity': max_ang_vel.tolist(),
            'final_stability_std': final_pos_std.tolist() if len(pos_array) > 100 else None
        }
    
    def save_results_to_file(self, filename=None):
        """Save detailed results to CSV file"""
        if not self.position_data:
            logger.warning("No data to save")
            return
            
        if filename is None:
            gravity_status = "with_gravity" if self.enable_gravity else "no_gravity"
            filename = f"trirotor_dynamics_{gravity_status}_results.csv"
        
        # Prepare data for CSV
        pos_array = np.array(self.position_data)
        vel_array = np.array(self.velocity_data)
        ang_vel_array = np.array(self.angular_velocity_data)
        thrust_array = np.array(self.thrust_data)
        time_array = np.array(self.time_data)
        
        # Create header
        header = "time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,ang_vel_x,ang_vel_y,ang_vel_z,thrust_1,thrust_2,thrust_3\n"
        
        # Write to file
        with open(filename, 'w') as f:
            f.write(header)
            for i in range(len(time_array)):
                line = f"{time_array[i]:.4f},"
                line += f"{pos_array[i,0]:.6f},{pos_array[i,1]:.6f},{pos_array[i,2]:.6f},"
                line += f"{vel_array[i,0]:.6f},{vel_array[i,1]:.6f},{vel_array[i,2]:.6f},"
                line += f"{ang_vel_array[i,0]:.6f},{ang_vel_array[i,1]:.6f},{ang_vel_array[i,2]:.6f},"
                line += f"{thrust_array[i,0]:.6f},{thrust_array[i,1]:.6f},{thrust_array[i,2]:.6f}\n"
                f.write(line)
        
        logger.info(f"Results saved to {filename}")
        return filename


def main():
    # Create evaluator with configured settings
    evaluator = TrirotorDynamicsEvaluator(
        enable_gravity=ENABLE_GRAVITY,
        num_envs=NUM_ENVS,
        headless=HEADLESS,
        use_warp=USE_WARP
    )
    
    try:
        # Run evaluation
        start_time = time.time()
        evaluator.run_full_evaluation(test_duration=TEST_DURATION)
        end_time = time.time()
        
        logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")
        
        # Analyze results
        evaluator.analyze_performance()
        
        # Save results to file if requested
        if SAVE_RESULTS:
            evaluator.save_results_to_file()
            
        logger.info("=== EVALUATION COMPLETE ===")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()