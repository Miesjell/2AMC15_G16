# Continuous PPO Agent for Gridworld

A complete implementation of Proximal Policy Optimization (PPO) for continuous state spaces with discrete actions, featuring real-time visualization and live training capabilities.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick training
python run.py train

# Live training with GUI
python run.py live

# Debug mode
python run.py debug

# List all available scripts
python run.py list
```

## 📁 Project Structure

```
├── agents/                    # RL Agent implementations
│   ├── continuous_ppo_agent.py  # Main PPO agent for continuous spaces
│   ├── base_agent.py           # Base agent interface
│   └── ...                     # Other agent implementations
├── world/                     # Environment implementations
│   ├── continuous_environment.py # Continuous gridworld environment
│   └── ...                     # Other environment implementations
├── demos/                     # Interactive demonstrations
│   ├── live_training_demo.py   # Live training with GUI
│   ├── realtime_demo.py        # Real-time agent visualization
│   └── demo_continuous_*.py    # Various demo scripts
├── examples/                  # Example and tutorial scripts
│   └── train_continuous_debug.py # Debug training script
├── training/                  # Training-specific scripts
│   ├── train_continuous.py     # Basic continuous training
│   └── ...                     # Other training variants
├── tests/                     # Test scripts
├── visualizations/           # Visualization outputs
├── grid_configs/             # Predefined grid configurations
├── train_main.py            # Main training script (recommended)
├── run.py                   # Project launcher script
└── requirements.txt         # Python dependencies
```

## 🎯 Key Features

### Continuous Environment
- **Continuous state space**: Agent position is (float, float)
- **Discrete action space**: 8-directional movement
- **Object-based collision detection**: Flexible object placement with radii
- **Distance-based rewards**: Guides agent toward targets
- **Real-time visualization**: Matplotlib and Pygame renderers

### PPO Agent
- **PyTorch implementation**: Modern deep learning framework
- **Adaptive exploration**: Epsilon-greedy with decay
- **Experience replay**: Efficient batch training
- **Gradient clipping**: Stable training
- **Model saving/loading**: Persistent training progress

### Visualization & Interaction
- **Live training GUI**: Watch agent learn in real-time
- **Interactive controls**: Pause, resume, reset training
- **Real-time metrics**: Episode rewards, success rates, training loss
- **Agent trail visualization**: See movement patterns
- **Training plots**: Comprehensive progress visualization

## 🏃‍♂️ Usage Examples

### Basic Training
```bash
# Quick training (200 episodes)
python train_main.py --quick

# Full training (1000 episodes)  
python train_main.py

# Custom configuration
python train_main.py --config my_config.json --episodes 500
```

### Live Training Demo
```bash
# With GUI
python demos/live_training_demo.py

# Without GUI (faster)
python demos/live_training_demo.py --no-gui

# Custom grid
python demos/live_training_demo.py --grid grid_configs/large_grid.npy
```

### Debug Mode
```bash
# Debug training with detailed output
python examples/train_continuous_debug.py --episodes 100

# Quiet mode
python examples/train_continuous_debug.py --quiet
```

## ⚙️ Configuration

The training system uses JSON configuration files. Example:

```json
{
  "agent": {
    "lr": 0.001,
    "gamma": 0.99,
    "eps_clip": 0.2,
    "epsilon": 0.15,
    "batch_size": 32,
    "buffer_size": 512
  },
  "training": {
    "episodes": 1000,
    "max_steps": 300,
    "eval_frequency": 50
  },
  "environment": {
    "grid_size": [8, 8],
    "step_size": 0.1,
    "start_pos": [1.5, 1.5],
    "target_pos": [6.5, 6.5]
  }
}
```

## 🎮 Controls (Live Demo)

- **SPACE**: Pause/Resume training
- **R**: Reset current episode  
- **T**: Toggle fast mode
- **ESC**: Quit training

## 📊 Monitoring Training

The system provides comprehensive training metrics:

- **Episode rewards**: Track learning progress
- **Success rates**: Percentage of episodes reaching target
- **Training loss**: Monitor optimization progress
- **Episode lengths**: Efficiency of learned policy
- **Evaluation results**: Periodic assessment during training

Results are saved to timestamped directories with:
- Training plots (PNG)
- Metrics data (JSON)
- Model checkpoints (PTH)
- Configuration used (JSON)

## 🔧 Customization

### Custom Reward Functions
```python
def my_reward_function(grid, new_pos, old_pos):
    # Your custom reward logic
    return reward

env = ContinuousEnvironment(
    grid_fp="my_grid.npy",
    reward_fn=my_reward_function
)
```

### Custom Agent Parameters
```python
agent = ContinuousPPOAgent(
    state_bounds=((0.0, 10.0), (0.0, 10.0)),
    lr=0.001,
    gamma=0.95,
    eps_clip=0.2,
    # ... other parameters
)
```

### Custom Grid Layouts
Create numpy arrays with:
- `0`: Empty space
- `1`: Wall/Obstacle  
- `3`: Target location

```python
import numpy as np
grid = np.zeros((10, 10))
grid[0, :] = 1    # Top wall
grid[:, 0] = 1    # Left wall
grid[8, 8] = 3    # Target
np.save("my_grid.npy", grid)
```

## 🐛 Troubleshooting

### Agent Not Learning
1. Check reward function - ensure targets are marked as `3` in grid
2. Adjust exploration parameters (epsilon, temperature)
3. Reduce training frequency (update less often)
4. Verify environment bounds match agent state_bounds

### GUI Issues
```bash
# Install pygame if missing
pip install pygame

# Run without GUI if needed
python run.py live --no-gui
```

### Performance Issues
- Reduce grid size for faster training
- Use `--quick` flag for shorter training
- Enable fast mode in live demo (press 'T')

## 📈 Results

Expected performance with default settings:
- **Convergence**: 200-500 episodes
- **Success rate**: 80%+ after training
- **Episode length**: Decreases from max_steps to ~50 steps

The agent learns to navigate from start to target efficiently while avoiding obstacles.

## 🤝 Contributing

This is a educational/research project. Feel free to:
- Experiment with different architectures
- Try alternative reward functions
- Add new visualization features
- Implement other RL algorithms

## 📝 Original Project

This work is based on the [Data Intelligence Challenge 2AMC15-2025](https://github.com/DataIntelligenceChallenge/2AMC15-2025) with significant extensions for continuous environments and PPO implementation.

## 🔗 References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenAI Gym](https://gym.openai.com/)
