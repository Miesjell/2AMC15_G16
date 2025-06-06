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
  - Periodically evaluates and records agent performance
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
  note: skip this this step if using [uv](https://docs.astral.sh/uv/)
   ```bash
   conda create -n dic2025 python=3.11
   conda activate dic2025
   ```
  

2. **Clone the repository**:
   ```bash
  Wasn't anonymous hihi
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   or alternatively using [uv](https://docs.astral.sh/uv/)
   ```bash
   uv sync
   ```

4. **Train a specific agent**:
   ```bash
    usage: train.py [-h] [--no_gui] [--sigma SIGMA] [--fps FPS] [--iter ITER]
                [--random_seed RANDOM_SEED] 
                GRID [GRID ...]

    DIC Reinforcement Learning Trainer.

    positional arguments:
    GRID                  Paths to the grid file to use. There can be more than
                        one.
    options:
    -h, --help                 show this help message and exit
    --no_gui                   Disables rendering to train faster (boolean)
    --sigma SIGMA              Sigma value for the stochasticity of the environment. (float, default=0.1, should be in [0, 1])
    --fps FPS                  Frames per second to render at. Only used if no_gui is not set. (int, default=30)
    --iter ITER                Number of iterations to go through. Should be integer. (int, default=1000)
    --random_seed RANDOM_SEED  Random seed value for the environment. (int, default=0)
    --episodes                 Number of training loops to go through. Should be integer. (int, default=501)
    ```

   This will generate:
   - Console output of the training process
   - A CSV log and a PNG learning curve under `learning_curves/`

5. **Run all agents and compare**:
   ```bash
   python compare.py
   ```
    This will generate:
   - Comparable learning curves of all three algorithms under `learning_curves/`
   
   You can adjust settings directly in `compare.py`, such as:
   ```python
   grid_path = "grid_configs/A1_grid.npy"
   episodes = 501
   sigma = 0.1
   ```
---
## Code guide

The code is made up of 2 modules: 

1. `agent`
2. `world`

### The `agent` module

The `agent` module contains the `BaseAgent` class as well as some benchmark agents you may want to test against.

The `BaseAgent` is an abstract class and all RL agents for DIC must inherit from/implement it.
If you know/understand class inheritence, skip the following section:

#### `BaseAgent` as an abstract class
Here you can find an explanation about abstract classes [Geeks for Geeks](https://www.geeksforgeeks.org/abstract-classes-in-python/).

Think of this like how all models in PyTorch start like 

```python
class NewModel(nn.Module):
    def __init__(self):
        super().__init__()
    ...
```

In this case, `NewModel` inherits from `nn.Module`, which gives it the ability to do back propagation, store parameters, etc. without you having to manually code that every time.
It also ensures that every class that inherits from `nn.Module` contains _at least_ the `forward()` method, which allows a forward pass to actually happen.

In the case of your RL agent, inheriting from `BaseAgent` guarantees that your agent implements `update()` and `take_action()`.
This ensures that no matter what RL agent you make and however you code it, the environment and training code can always interact with it in the same way.
Check out the benchmark agents to see examples.

### The `world` module

The world module contains:
1. `grid_creator.py`
2. `environment.py`
3. `grid.py`
4. `gui.py`

#### Grid creator
Run this file to create new grids.

```bash
$ python grid_creator.py
```

This will start up a web server where you create new grids, of different sizes with various elements arrangements.
To view the grid creator itself, go to `127.0.0.1:5000`.
All levels will be saved to the `grid_configs/` directory.


#### The Environment

The `Environment` is very important because it contains everything we hold dear, including ourselves [^1].
It is also the name of the class which our RL agent will act within. Most of the action happens in there.

The main interaction with `Environment` is through the methods:

- `Environment()` to initialize the environment
- `reset()` to reset the environment
- `step()` to actually take a time step with the environment
- `Environment().evaluate_agent()` to evaluate the agent after training.

[^1]: In case you missed it, this sentence is a joke. Please do not write all your code in the `Environment` class.

#### The Grid

The `Grid` class is the the actual representation of the world on which the agent moves. It is a 2D Numpy array.

#### The GUI

The Graphical User Interface provides a way for you to actually see what the RL agent is doing.
While performant and written using PyGame, it is still about 1300x slower than not running a GUI.
Because of this, we recommend using it only while testing/debugging and not while training.

##  Experiments Instruction

You are encouraged to test how different setups affect learning:
- Different **grid maps**: swap in other `.npy` files under `grid_configs/`
- **Discount factor** `gamma`: change this within the agent definition `agents/`
- **Stochasticity** `sigma`: pass via `--sigma` argument to `train.py` or in `compare.py`
- **Episode count**: pass via `--episodes` argument to `train.py` or in `compare.py`
- **Reward Function**: uncomment the reward function that you want in `world/environment.py`
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
environment.py   # Modified reward function
```

---
