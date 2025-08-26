# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aerial Gym Simulator is a high-fidelity physics-based simulator for training Micro Aerial Vehicle (MAV) platforms built on NVIDIA Isaac Gym. It supports thousands of simultaneous multirotor simulations with GPU-accelerated geometric controllers and custom ray-casting sensors.

## Installation and Setup

The project uses Python with Isaac Gym as the core physics engine. Install dependencies using:
```bash
pip install -e .
```

Key dependencies include:
- isaacgym (NVIDIA Isaac Gym)
- warp-lang==1.0.0 (custom rendering framework)
- torch, pytorch3d (neural networks)
- rl-games, sample-factory (reinforcement learning)
- urdfpy, trimesh (robot model handling)
- numpy==1.23 (specific version required)

## Common Development Commands

### Running Examples
```bash
# Basic position control simulation
python aerial_gym/examples/position_control_example.py --num_envs=64 --headless

# Navigation task example
python aerial_gym/examples/navigation_task_example.py --use_warp

# RL training with rl-games
python aerial_gym/rl_training/rl_games/runner.py --task=quad --num_envs=4096

# RL training with sample-factory
python aerial_gym/rl_training/sample_factory/aerialgym_examples/train_aerialgym.py
```

### Testing and Validation
No formal test suite is present - validation is done through running example scripts and verifying simulation behavior.

### Code Formatting
Uses Black formatter with 100 character line length (configured in pyproject.toml).

## Architecture Overview

### Core Components

**SimBuilder (`aerial_gym/sim/sim_builder.py`)**
- Central factory for creating simulation environments
- Builds environments by combining sim, env, robot, and controller configurations
- Returns EnvManager instance for simulation control

**EnvManager (`aerial_gym/env_manager/env_manager.py`)**
- Main simulation orchestrator 
- Handles environment resets, stepping, and state management
- Interfaces with Isaac Gym physics engine

**Registry System (`aerial_gym/registry/`)**
- Component registry for tasks, environments, robots, controllers
- Enables modular configuration and easy component swapping
- Each registry maintains class and config mappings

### Configuration Architecture

All components use a hierarchical configuration system:
- `aerial_gym/config/` contains all configuration classes
- Configs are organized by component type (robot, env, task, sensor, etc.)
- Each config inherits from a base config class
- Configs define parameters like physics properties, sensor specs, control gains

### Key Component Types

**Robots (`aerial_gym/robots/`)**
- Base classes: `BaseMultirotor`, `BaseROV`, `BaseReconfigurable`  
- Specific robots: quadrotors, octarotors, underwater vehicles, flexible arms
- Each robot has corresponding URDF models in `resources/robots/`

**Controllers (`aerial_gym/control/controllers/`)**
- GPU-accelerated geometric controllers
- Types: position, velocity, attitude, acceleration control
- Lee controller implementation for multirotors
- Fully-actuated control for complex platforms

**Tasks (`aerial_gym/task/`)**
- Define objectives and reward functions
- Examples: position setpoint, navigation, sim2real transfer
- Tasks inherit from `BaseTask` and implement reset/step logic

**Sensors (`aerial_gym/sensors/`)**
- IMU sensors with realistic noise models
- Camera sensors (Isaac Gym and custom Warp-based)
- LiDAR sensors with configurable properties
- Warp-based sensors enable high-speed parallel rendering

**Environments (`aerial_gym/config/env_config/`)**
- Scene definitions with obstacles and objects
- Dynamic environments with moving obstacles
- Forest environments for navigation training

### Warp Integration

Custom rendering framework built on NVIDIA Warp:
- `aerial_gym/sensors/warp/` contains Warp-based sensors
- Enables custom sensor implementations and parallel operations
- Used for depth cameras, segmentation, and LiDAR simulation
- Kernel-based operations in `warp_kernels/`

## Development Guidelines

### Adding New Components

1. Create config class in appropriate `config/` subdirectory
2. Implement component class inheriting from base class
3. Register component in corresponding registry
4. Add URDF/asset files to `resources/` if needed
5. Create example script demonstrating usage

### Working with Configurations

- Configs are Python classes, not YAML files
- Override specific parameters by subclassing base configs  
- Use registry system to reference configs by name strings
- Configs are passed to component constructors during initialization

### Simulation Parameters

- Default timestep: 2ms (configurable via sim configs)
- Number of environments: configurable, typically 64-4096 for training
- GPU memory management: call `torch.cuda.empty_cache()` when deleting environments
- Use `headless=True` for training, `False` for visualization

### File Organization

- Example scripts in `aerial_gym/examples/`
- RL training scripts in `aerial_gym/rl_training/`
- Pre-trained models in `aerial_gym/rl_training/*/networks/`
- Asset files (URDF, meshes) in `resources/`
- Sim2real utilities in `aerial_gym/sim2real/`

The codebase follows a modular design where components can be mixed and matched through the configuration and registry system, enabling rapid prototyping of new robots, environments, and tasks.