# Quadruped Robot Learning: Adaptive Locomotion

A complete reinforcement learning pipeline for training a Unitree Go1 quadruped robot to walk adaptively across varied terrains using MuJoCo physics simulation and PPO.

## Overview

This project trains a simulated quadruped robot (Unitree Go1) to learn locomotion behaviors from scratch using Proximal Policy Optimization (PPO). The robot learns to:

- Walk forward with a stable trotting gait
- Maintain balance and recover from perturbations
- Navigate varied terrains (rough ground, stairs, slopes, gaps)
- Track commanded velocities (forward, lateral, turning)

## Project Structure

```
mujoco-quadruped/
├── envs/
│   ├── quadruped_env.py          # Custom Gymnasium environment
│   └── terrain_generator.py      # Procedural terrain generation
├── models/
│   ├── go1.xml                   # Unitree Go1 robot model (from MuJoCo Menagerie)
│   ├── scene.xml                 # Default flat-ground scene
│   ├── assets/                   # Robot mesh files (.stl)
│   └── terrain_*.xml             # Generated terrain scenes
├── training/
│   ├── train.py                  # Main PPO training script
│   ├── config.yaml               # Hyperparameters and settings
│   └── callbacks.py              # Video recording, reward logging, curriculum
├── evaluation/
│   ├── evaluate.py               # Multi-terrain evaluation
│   └── visualize.py              # Video generation and training plots
├── trained_models/               # Saved model checkpoints
├── logs/                         # TensorBoard training logs
├── videos/                       # Recorded demo videos
├── requirements.txt
├── setup.py                      # Setup verification script
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python setup.py
```

### 3. Train

```bash
# Basic training (1M steps, ~3 hours on CPU)
python training/train.py --config training/config.yaml

# Resume from checkpoint
python training/train.py --config training/config.yaml --resume trained_models/go1_ppo_500000_steps.zip

# Monitor with TensorBoard
tensorboard --logdir logs/
```

### 4. Evaluate

```bash
# Evaluate on all terrains
python evaluation/evaluate.py trained_models/go1_ppo_final.zip --terrain all

# Evaluate on specific terrain
python evaluation/evaluate.py trained_models/go1_ppo_final.zip --terrain rough --n-episodes 50
```

### 5. Generate Videos

```bash
# Record demos on all terrains
python evaluation/visualize.py videos trained_models/go1_ppo_final.zip --comparison

# Plot training curves
python evaluation/visualize.py plots --log-dir logs/

# Plot evaluation comparison
python evaluation/visualize.py eval-plots evaluation_results.json
```

## Environment Design

### Observation Space (56 dimensions)

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Joint positions | 12 | Offset from default standing pose |
| Joint velocities | 12 | Scaled joint angular velocities |
| Trunk quaternion | 4 | Body orientation |
| Trunk angular velocity | 3 | Body rotational velocity |
| Trunk linear velocity | 3 | Body translational velocity |
| Foot contacts | 4 | Binary ground contact per foot |
| Gravity vector | 3 | Gravity direction in body frame |
| Commanded velocity | 3 | Target [vx, vy, yaw_rate] |
| Previous actions | 12 | Last action for smoothness |

### Action Space (12 dimensions)

Target joint position offsets (scaled by `action_scale=0.3`) added to the default standing pose. The Go1 has 3 joints per leg (hip abduction, thigh, knee) x 4 legs = 12 actuators.

### Reward Function

```
reward = velocity_tracking_weight * exp(-2 * velocity_error)  # Track commanded velocity
       + alive_bonus                                           # Stay alive
       - ctrl_cost_weight * ||action||²                        # Energy efficiency
       - smoothness_weight * ||action - prev_action||²         # Smooth motions
       - orientation_weight * ||body_tilt||²                   # Stay upright
       - height_weight * (height - target)²                    # Maintain height
       - foot_slip_weight * slip_speed                         # Minimize foot slip
```

## Terrain Types

The terrain generator creates 7 terrain variants:

| Terrain | Description |
|---------|-------------|
| **Flat** | Baseline flat ground |
| **Rough** | Random cylindrical bumps (up to 4cm) |
| **Stairs Up** | Ascending staircase (4cm steps) |
| **Stairs Down** | Descending staircase |
| **Slope** | 10-degree inclined surface |
| **Gaps** | Platforms with 15cm gaps |
| **Mixed** | Combination of bumps, steps, and gaps |

## Training Details

### Algorithm: PPO (Proximal Policy Optimization)

**Why PPO?**
- Stable training with monotonic improvement guarantees
- Works well with continuous action spaces
- Parallelizable across multiple environments
- Good sample efficiency for locomotion tasks

### Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 3e-4 | Standard for PPO locomotion |
| Batch size | 64 | Balanced gradient estimates |
| Rollout length | 2048 | ~40 episodes worth of data |
| Epochs | 10 | Multiple passes over rollout |
| Gamma | 0.99 | Long-horizon locomotion |
| GAE lambda | 0.95 | Variance-bias tradeoff |
| Clip range | 0.2 | Conservative policy updates |
| Entropy coeff | 0.01 | Exploration encouragement |
| Network | [256, 256] ELU | Sufficient capacity for locomotion |

### Domain Randomization

When enabled, randomizes during training:
- Body mass: ±15%
- Joint damping: ±20%
- Ground friction: ±30%
- Motor strength: ±10%

### Curriculum Learning

Optional staged training progression:
1. **Flat basic**: Slow walking on flat ground
2. **Flat fast**: Faster speeds on flat ground
3. **Rough terrain**: Walking on uneven surfaces with domain randomization
4. **Mixed terrain**: All terrain types with full randomization

## Technical Stack

| Package | Version | Purpose |
|---------|---------|---------|
| MuJoCo | 3.4.0 | Physics simulation |
| Gymnasium | 1.2.3 | RL environment interface |
| Stable-Baselines3 | 2.7.1 | PPO implementation |
| PyTorch | 2.10.0 | Neural network backend |
| NumPy | 2.4.2 | Numerical computation |
| Matplotlib | 3.10.8 | Plotting and analysis |
| TensorBoard | 2.20.0 | Training monitoring |
| ImageIO | 2.37.2 | Video recording |

## Design Decisions

### Position Control vs Torque Control
We use **position control** (PD controllers) rather than raw torque control. This matches the real Go1's control interface and is more stable for RL training, as the PD controller provides implicit damping.

### Observation Design
The observation includes the **gravity vector in body frame** rather than raw euler angles, avoiding gimbal lock issues. **Previous actions** are included to encourage smooth motions and make the policy aware of its recent behavior.

### Reward Shaping
The reward uses **exponential tracking** (`exp(-2 * error²)`) for velocity tracking rather than linear rewards. This provides a smooth, bounded signal that encourages precise tracking without reward explosion.

### Sim-to-Real Considerations
Domain randomization of physical parameters (mass, friction, motor strength) helps bridge the sim-to-real gap. The observation space is designed to be available on real hardware (joint encoders, IMU, foot contact sensors).

## Extensions

- **Multi-algorithm comparison**: Train with SAC, TD3, and compare gaits
- **Vision-based navigation**: Add camera observations for obstacle avoidance
- **Trajectory optimization**: MPC baseline for comparison
- **Real robot deployment**: Transfer to physical Unitree Go1
- **Interactive control**: Real-time velocity commands via keyboard/gamepad

## License

The Unitree Go1 model is from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache 2.0).
