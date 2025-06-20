# Car Racing Reinforcement Learning

A comprehensive implementation of reinforcement learning agents for the OpenAI Gymnasium CarRacing-v3 environment. This project explores two state-of-the-art RL algorithms: Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) to train autonomous racing agents.

## Project Overview

This repository implements AI agents that learn to navigate a car around randomly generated racing tracks. The agents use computer vision to process track images and make real-time driving decisions, optimizing for track completion while minimizing time penalties.

### Key Features

- **Two RL Algorithms**: DQN with Double Q-Learning and PPO with parallel environments
- **Advanced Training Techniques**: Experience replay, curriculum learning, and mixed precision training
- **Performance Optimizations**: CUDA/MPS support, model compilation, and parallel processing
- **Comprehensive Monitoring**: Weights & Biases integration for experiment tracking
- **Robust Evaluation**: Dedicated testing framework with performance metrics

## Environment

The project uses the **CarRacing-v3** environment from OpenAI Gymnasium:
- **Observation Space**: 96×96×3 RGB images of the track
- **Action Space**: Continuous control (discretized to 8-11 actions)
- **Objective**: Complete laps while minimizing time penalties

## Algorithms Implemented

### 1. Deep Q-Network (DQN) - `rl_implementation.py`

A sophisticated DQN implementation with advanced features:

- **Neural Architecture**: 3-layer CNN with batch normalization + fully connected layers
- **Experience Replay**: 300k capacity buffer for stable learning
- **Double DQN**: Reduces overestimation bias in Q-value learning
- **Curriculum Learning**: Gradually increases episode length (200→1000 steps)
- **Exploration Strategy**: Adaptive epsilon-greedy with decay
- **Action Space**: 11 discrete actions (steering + gas/brake combinations)

### 2. Proximal Policy Optimization (PPO) - `rl_ppo.py`

A modern policy gradient method with actor-critic architecture:

- **Parallel Training**: 8 simultaneous environments for faster learning
- **Actor-Critic Design**: Shared CNN backbone with separate policy/value heads
- **Performance Optimizations**: Mixed precision training and model compilation
- **Advantage Estimation**: Generalized Advantage Estimation (GAE)
- **Action Space**: 8 discrete actions covering key driving maneuvers

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd car_racing_rl
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up Weights & Biases** (optional but recommended):
```bash
wandb login
```

## Quick Start

### Manual Control (Test Environment)
```bash
python main.py
```
Use arrow keys to control the car and understand the environment dynamics.

### Training Agents

**Train DQN Agent**:
```bash
python rl_implementation.py
```

**Train PPO Agent**:
```bash
python rl_ppo.py
```

Training will automatically:
- Save checkpoints to timestamped directories
- Log metrics to Weights & Biases
- Track best performing models

### Evaluate Trained Models

```bash
python test.py --model_path path/to/model.pth --algorithm dqn --episodes 10 --render
```

Options:
- `--algorithm`: Choose between 'dqn' or 'ppo'
- `--episodes`: Number of evaluation episodes
- `--render`: Enable visual rendering
- `--deterministic`: Use deterministic policy (for PPO)

## Project Structure

```
car_racing_rl/
├── main.py                 # Manual control interface
├── rl_implementation.py    # DQN implementation
├── rl_ppo.py              # PPO implementation  
├── test.py                # Model evaluation script
├── requirements.txt       # Dependencies
├── notes/                 # Documentation
│   ├── rl_implementation.qmd
│   └── residual_networks.qmd
└── checkpoints/           # Saved models (created during training)
```

## Key Implementation Details

### Preprocessing Pipeline
- RGB → Grayscale conversion
- Resize to 96×96 pixels
- Normalization to [0,1] range
- Frame stacking for temporal information

### Reward Engineering
- Track completion bonuses
- Time penalty minimization
- Reward normalization for stable learning

### Training Stability Features
- Gradient clipping
- Batch normalization
- Smooth L1 loss (DQN)
- Entropy regularization (PPO)

## Hardware Requirements

### Minimum
- CPU: Multi-core processor
- RAM: 8GB+
- GPU: Optional but recommended

### Recommended
- GPU: NVIDIA RTX 3060+ or Apple Silicon M1/M2
- RAM: 16GB+
- CUDA 11.0+ (for NVIDIA GPUs)

The implementation automatically detects and utilizes available hardware acceleration (CUDA/MPS).

## Monitoring and Logging

The project integrates with Weights & Biases for comprehensive experiment tracking:

- Real-time loss and reward curves
- Episode statistics and performance metrics
- Model checkpointing and versioning
- Hyperparameter tracking

## Results and Performance

Trained agents demonstrate:
- Successful track completion on various track layouts
- Smooth cornering and speed optimization
- Robust performance across different environments
- Convergence within reasonable training time

## File Descriptions

- **`main.py`**: Human-playable version for environment familiarization
- **`rl_implementation.py`**: Complete DQN implementation with curriculum learning
- **`rl_ppo.py`**: PPO implementation with parallel environments
- **`test.py`**: Evaluation framework with flexible testing options
- **`notes/`**: Detailed documentation and implementation explanations

## Contributing

This project serves as a comprehensive example of modern reinforcement learning techniques applied to autonomous driving. The implementations include extensive comments and documentation for educational purposes.

## License

This project is intended for educational and research purposes.

## Acknowledgments

- OpenAI Gymnasium for the CarRacing environment
- PyTorch team for the deep learning framework
- Weights & Biases for experiment tracking tools