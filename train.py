"""
Train your RL Agent in this file.
"""

from argparse import ArgumentParser
import importlib
from pathlib import Path
import re
from tqdm import trange

try:
    from world import Environment
    from agents.random_agent import RandomAgent
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys
    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.extend(root_path)
    from world import Environment
    from agents.random_agent import RandomAgent

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if no_gui is not set.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("-a", "--agent", type=str, default="RandomAgent",
                   help="Name of the agent class to use (e.g., RandomAgent)")
    return p.parse_args()

def load_agent(agent_name: str):
    """
    Dynamically load and instantiate an agent class based on its name.
    """
    try:
        module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', agent_name).lower()
        module = importlib.import_module(f"agents.{module_name}")
        agent_class = getattr(module, agent_name)
        print(f"Loaded agent class: {agent_class}")
        return agent_class()
    except (ImportError, AttributeError) as e:
        print(f"Error loading agent '{agent_name}': {e}")
        print("Falling back to RandomAgent.")
        return RandomAgent()

def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, agent_name: str):

    for grid in grid_paths:

        # Set up the environment with fixed start position for fair comparison
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
                          random_seed=random_seed, agent_start_pos=(3, 11))

        # Initialize agent
        agent = load_agent(agent_name)
        print(f"Agent: {agent}")

        # Reset environment to get initial state
        state = env.reset()
        for _ in trange(iters):

            # Agent takes action based on current state
            action = agent.take_action(state)

            # Take action in the environment
            next_state, reward, terminated, info = env.step(action)

            # Inform agent of the result (update policy/values/etc.)
            agent.update(next_state, reward, info["actual_action"], info)

            # End of episode logic for episodic agents (e.g., MC)
            if terminated:
                if hasattr(agent, "update_Q"):
                    agent.update_Q()
                break

            state = next_state  # move to next state

        # Evaluate trained agent
        Environment.evaluate_agent(grid, agent, iters, sigma,
                                   random_seed=random_seed,
                                   agent_start_pos=(3, 11),
                                   show_images=False)

if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed, args.agent)
