# Extended Reinforcement Learning Agents — DIC 2AMC15-2025 Project

This repository is based on the official [Data Intelligence Challenge 2AMC15-2025 repository](https://github.com/DataIntelligenceChallenge/2AMC15-2025) and contains our extended work for exploring and comparing various reinforcement learning algorithms in a grid world environment.

##  What's New

We extended the original environment with the following contributions:

- **New agents** implemented under the `agents/` folder:
  - `MonteCarloAgent`
  - `QLearningAgent`
  - `ValueIterationAgent`
- **Modified `train.py`**:
  - Added support for extra parameters like `--episodes` for training length.
  - Exports **learning curve data** as CSV and plots PNG images under `learning_curves/`.
  - Automatically selects appropriate start positions based on the grid file
  - Periodically evaluates and records agent performance (every 50 episodes)
- **New script: `compare.py`**:
  - Automates training all three agents.
  - Produces a combined learning curve plot for visual comparison.
- **Modified  `environment.py`**
  - Customizable **reward function**
- **Experimentation-ready**:
  - Configurable grid environments (e.g., `Open_Field.npy`).
  - Adjustable parameters like discount factor `gamma`, stochasticity `sigma`, and `episodes`.

---

##  Quickstart (with extensions)

1. **Set up the environment**:
   ```bash
   conda create -n dic2025 python=3.11
   conda activate dic2025
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/Miesjell/2AMC15_G16.git
   cd 2AMC15_G16
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train a specific agent**:
   ```bash
   python train.py grid_configs/A1_grid.npy --agent QLearningAgent --episodes 500 --sigma 0.1
   ```

   This will generate:
   - Console output of the training process
   - A CSV log and a PNG learning curve under `learning_curves/`
   - The modified `train.py` supports additional command-line arguments beyond the original

5. **Run all agents and compare**:
   ```bash
   python compare.py
   ```

   You can adjust settings directly in `compare.py`, such as:
   ```python
   grid_path = "grid_configs/A1_grid.npy"
   episodes = 501
   sigma = 0.1
   ```
---


##  Experiments Instruction

You are encouraged to test how different setups affect learning:
- Different **grid maps**: swap in other `.npy` files under `grid_configs/`
- **Discount factor** `gamma`: change this within the agent definition `agents/`
- **Stochasticity** `sigma`: pass via `--sigma` argument to `train.py` or in `compare.py`
- **Episode count**: pass via `--episodes` argument to `train.py` or in `compare.py`
- **Reward Function**: modify the `reward_fn` in `environment.py`
---

##  Project Structure (New Files Only)

```
agents/
├── MonteCarloAgent.py
├── QLearningAgent.py
├── ValueIterationAgent.py

learning_curves/
├── QLearningAgent_curve.csv
├── compare_learning_curves_*.png

compare.py  # Script to train and compare all agents
train.py    # Modified to accept more parameters and plot learning curves
environment.py   # Modified to have new reward function
```

---
