# Extended Reinforcement Learning Agents — DIC 2AMC15-2025 Project

This repository is based on the official [Data Intelligence Challenge 2AMC15-2025 repository](https://github.com/DataIntelligenceChallenge/2AMC15-2025) and contains our extended work for exploring and comparing various reinforcement learning algorithms in a grid world environment.

##  What's New

We extended the original environment with the following contributions:

- **New agents** implemented under the `agents/` folder:
  - `DQNAgent`
  - `PPOAgent`
- **New scirpt `train_dqn.py, train_ppo.py`**:
  - Added support for extra parameters like `--episodes` for training length.
  - Exports **learning curve data** as CSV and plots PNG images under `learning_curves/`.
  - Periodically evaluates and records agent performance
- **New script: `stats_agent.py, stats_sigma.py, stats_step_size.py`**:
  - Average return over the last 50 episodes.
  - It plots success rate (% of episodes where the agent reached the goal).
  - It plots average number of steps taken to reach the target.
  - It plots 1 learning curve per experiment.
- **Modified  `continiousEnvironment.py`**
  - Continuous agent movement with adjustable step size, agent_size.
  - Collision detection and reward handling, including penalties for walls and rewards for reaching the goal.
  - Modular reward function, customizable to encourage exploration and path efficiency
- **Experimentation-ready**:
  - Two grid environments (`mainrestaurant.npy`) and (`mainrestaurant2.npy`)
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
   git clone https://github.com/Miesjell/2AMC15_G16.git
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   or alternatively using [uv](https://docs.astral.sh/uv/)
   ```bash
   uv sync
   ```

4. **Train each agent**:
   ```bash
    1. For DQNAgent: python train_dqn.py grid_configs/mainrestaurant.npy [--no_gui]
    ```

    ```bash
     2. For PPOAgent: python train_ppo.py grid_configs/mainrestaurant.npy [--no_gui]
    ```
    positional arguments:

    ```bash
    GRID                  Paths to the grid file to use. There can be more than one.

    Options:
    -h, --help                 show this help message and exit
    --no_gui                   Disables rendering to train faster (boolean)
    --sigma SIGMA              Sigma value for the stochasticity of the environment. (float, default=0.1, should be in [0,1])
    --fps FPS                  Frames per second to render at. Only used if no_gui is not set. (int, default=30)
    --iter ITER                Number of iterations to go through. Should be integer. (int, default=1000)
    --random_seed RANDOM_SEED  Random seed value for the environment. (int, default=0)
    --episodes                 Number of training loops to go through. Should be integer. (int, default=3000)
    ```

   This will generate:
   - Console output of the training process
   - A CSV log and a PNG learning curve under `learning_curves/`

5. **Run all agents and compare**:
   ```bash
   python stats_sigma.py
   python stats_agent_size.py
   python stats_step_size.py
   ```
    This will generate:
   - Average return over the last 50 episodes.
   - Success rate (% of episodes where the agent reached the goal) plot.
   - Average number of steps taken to reach the target plot.
   - 1 learning curve plot per experiment.
   
   You can adjust settings directly in all of the 3 `stats.py`, such as:
   ```python
   episodes = load_episode_data([...])
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
├── DQNAgent.py
├── PPOAgent.py

results2/
├── summary_expiriment.csv
├── compare_succes_rates_agent_expiriment.png
├── compare_learning_curves_agent_experiment.png

traindqn.py   # Modified to accept more parameters and plot learning curves
trainppo.py  # Implemented to run based on the dqn agent.
continiousEnvironment.py   # Modified reward function and move fucntion...
stats_agent_size.py # post procressing of the train results for the experiments.
stats_sigma.py # post procressing of the train results for the experiments.
stats_step_size.py # post procressing of the train results for the experiments.
```

---
