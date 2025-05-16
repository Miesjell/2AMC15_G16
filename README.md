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
- **New script: `compare.py`**:
  - Automates training all three agents.
  - Produces a combined learning curve plot for visual comparison.
- **Experimentation-ready**:
  - Configurable grid environments (e.g., `A1_grid.npy`).
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

##  Modified Parameters

- Customizable **reward function**:
  - The default reward function has been modified to:
    - `-0.01` for moving to an empty tile
    - `-0.05` for hitting a wall or obstacle
    - `+5` for reaching a target
  - Users can pass their own reward function via the `reward_fn` parameter to the `Environment` constructor for custom experimentation.

The `train.py` script was significantly extended:

- Adds `--episodes` for training by episodes (rather than fixed iterations)
- Logs learning curves as both `.csv` and `.png` files in the `learning_curves/` folder
- Automatically selects appropriate start positions based on the grid file
- Periodically evaluates and records agent performance (every 50 episodes)

The modified `train.py` supports additional command-line arguments beyond the original repo:

```bash
usage: train.py [--agent AGENT] [--episodes EPISODES] [--sigma SIGMA] ...
```
---

##  Experimentation

You are encouraged to test how different setups affect learning:
- Different **grid maps**: swap in other `.npy` files under `grid_configs/`
- **Discount factor** `gamma`: change this within the agent definition
- **Stochasticity** `sigma`: pass via `--sigma` argument
- **Episode count**: change via `--episodes` or in `compare.py`

- **This part should be more specific, TBC**
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
```

---
