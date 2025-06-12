"""
Train your RL Agent in this file.
"""

from argparse import ArgumentParser
import importlib
from pathlib import Path
import re
import matplotlib.pyplot as plt
from tqdm import trange
from datetime import datetime
import csv
import numpy as np

try:
    # Only import the continuous environment and alias it as Environment
    from world.continuousEnvironment import ContinuousEnvironment as Environment
    from agents.random_agent import RandomAgent
except ModuleNotFoundError:
    import sys, os
    # Fix sys.path if needed
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world.continuousEnvironment import ContinuousEnvironment as Environment
    from agents.random_agent import RandomAgent

# Helper to load any agent by class name
def load_agent(agent_name: str, env):
    """
    Dynamically load and instantiate an agent class based on its name.
    """
    try:
        module_name = re.sub(r"(?<!^)(?=[A-Z])", "_", agent_name).lower()
        module = importlib.import_module(f"agents.{module_name}")
        agent_class = getattr(module, agent_name)
        print(f"Loaded agent class: {agent_class}")
        return agent_class(env)
    except (ImportError, AttributeError) as e:
        print(f"Error loading agent '{agent_name}': {e}")
        print("Falling back to RandomAgent.")
        return RandomAgent(env)

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument(
        "GRID",
        type=Path,
        nargs="+",
        help="Paths to the grid file to use. There can be more than one.",
    )
    p.add_argument("--no_gui", action="store_true", help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1, help="Action stochasticity")
    p.add_argument("--fps", type=int, default=30, help="Frames per second if GUI enabled")
    p.add_argument("--iter", type=int, default=1000, help="Max steps per episode")
    p.add_argument("--random_seed", type=int, default=0, help="Random seed for reproducibility")
    p.add_argument(
        "-a", "--agent", type=str, default="RandomAgent",
        help="Name of the agent class to use (e.g., RandomAgent)"
    )
    p.add_argument("-e", "--episodes", type=int, default=1, help="Number of training episodes")
    return p.parse_args()

def main(
    grid_paths: list[Path],
    no_gui: bool,
    iters: int,
    fps: int,
    sigma: float,
    random_seed: int,
    agent_name: str,
    episodes: int,
):
    """Main loop of the program."""
    for grid in grid_paths:
        # Determine agent start position per grid
        startPos = None
        if grid.name == "mainrestaurant.npy":
            startPos = [8, 2]
        # Add other grid-specific start positions as before...

        # Instantiate the ContinuousEnvironment
        env = Environment(
            grid_fp=grid,
            no_gui=no_gui,
            sigma=sigma,
            target_fps=fps,
            random_seed=random_seed,
            agent_start_pos=startPos,
        )

        # Load the chosen agent
        agent = load_agent(agent_name, env)
        print(f"Agent: {agent}")

        episode_numbers = []
        episode_returns = []

        # Monte Carlo branch (untouched)
        if agent_name == "MC_Agent":
            for episode in trange(episodes, desc="Training episodes"):
                state = env.reset(agent_start_pos=startPos)
                episode_data = []
                step_count = 0
                while True:
                    action = agent.take_action(state)
                    next_state, reward, done, info = env.step(action)
                    episode_data.append((state, info["actual_action"], reward))
                    step_count += 1
                    if done or step_count >= getattr(agent, "max_episode_len", 3000):
                        agent.update(episode_data)
                        break
                    state = next_state
                agent.epsilon = max(0.05, agent.epsilon * 0.995)

                # Periodic evaluation using ContinuousEnvironment.evaluate_agent
                if episode % 50 == 0:
                    total_return = Environment.evaluate_agent(
                        grid_fp=grid,
                        agent=agent,
                        max_steps=iters,
                        sigma=sigma,
                        agent_start_pos=startPos,
                        random_seed=random_seed,
                        show_images=False
                    )
                    episode_numbers.append(episode + 1)
                    episode_returns.append(total_return)

            # Plot & save learning curve
            plt.plot(episode_numbers, episode_returns, label="Episode Return")
            plt.xlabel("Episode")
            plt.ylabel("Total Return")
            plt.title(f"{agent_name} on {grid.stem}")
            plt.grid(True)
            curve_dir = Path("learning_curves") / grid.stem
            curve_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(curve_dir / f"{grid.stem}_{timestamp}.png")
            with open(curve_dir / f"{agent_name}_curve.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "return"])
                writer.writerows(zip(episode_numbers, episode_returns))

        # Tabular Q-learning branch (unchanged)
        elif agent_name == "QLearningAgent":
            for episode in trange(episodes, desc="Training episodes"):
                state = env.reset()
                agent.reset()
                for _ in range(iters):
                    action = agent.take_action(state)
                    next_state, reward, done, info = env.step(action)
                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        break

        # DQN / PPO and other continuous-state agents
        else:
            for episode in trange(episodes, desc="Training episodes"):
                state = env.reset()
                if hasattr(agent, "reset"):
                    agent.reset()
                for _ in range(iters):
                    action = agent.take_action(state)
                    state, reward, done, info = env.step(action)
                    agent.update(state, reward, info["actual_action"])
                    if done:
                        break

                # Periodic evaluation using ContinuousEnvironment.evaluate_agent
                if episode % 50 == 0:
                    total_return = Environment.evaluate_agent(
                        grid_fp=grid,
                        agent=agent,
                        max_steps=iters,
                        sigma=sigma,
                        agent_start_pos=startPos,
                        random_seed=random_seed,
                        show_images=False
                    )
                    episode_numbers.append(episode + 1)
                    episode_returns.append(total_return)
                # Reset agent memory if necessary
                if hasattr(agent, "prev_state"):
                    agent.prev_state = None
                    agent.prev_action = None

            # Plot & save learning curve
            plt.plot(episode_numbers, episode_returns, label="Episode Return")
            plt.xlabel("Episode")
            plt.ylabel("Total Return")
            plt.title(f"{agent_name} on {grid.stem}")
            plt.grid(True)
            curve_dir = Path("learning_curves") / grid.stem
            curve_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(curve_dir / f"{grid.stem}_{timestamp}.png")
            with open(curve_dir / f"{agent_name}_curve.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "return"])
                writer.writerows(zip(episode_numbers, episode_returns))


if __name__ == "__main__":
    args = parse_args()
    main(
        args.GRID,
        args.no_gui,
        args.iter,
        args.fps,
        args.sigma,
        args.random_seed,
        args.agent,
        args.episodes,
    )
