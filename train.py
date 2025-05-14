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
    p.add_argument("--iter", type=int, default=10000,
                   help="Number of episodes to run during training.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("-a", "--agent", type=str, default="RandomAgent",
                   help="Name of the agent class to use (e.g., OnPolicyMonteCarlo)")
    return p.parse_args()

def load_agent(agent_name: str):
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

# Custom reward function
def goal_state_reward(grid, new_pos):
    """
    State-based reward:
    - Goal (3): +1000
    - Wall or obstacle (1 or 2): -5
    - Empty cell: -1
    """
    cell = grid[new_pos]
    if cell == 3:
        return 1000
    elif cell in (1, 2):
        return -5
    return -1

def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, agent_name: str, agent_start_pos=(3, 11)):

    for grid in grid_paths:
        # Training environment setup
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
                          random_seed=random_seed, agent_start_pos=agent_start_pos,
                          reward_fn=lambda grid, new_pos: goal_state_reward(grid, new_pos))

        agent = load_agent(agent_name)
        print(f"Agent: {agent}")

        # Training loop over episodes
        for episode in trange(iters, desc="Training Episodes"):
            state = env.reset(agent_start_pos=agent_start_pos)

            while True:
                action = agent.take_action(state)
                next_state, reward, terminated, info = env.step(action)
                agent.update(state, next_state, reward, action, info)

                # Break if environment is terminal OR agent signals episode end (e.g., max steps)
                if terminated or getattr(agent, "episode_done", False):
                    agent.episode_done = False  # Reset for next episode
                    break

                state = next_state

        # Freeze exploration before evaluation
        if hasattr(agent, "freeze_policy"):
            agent.freeze_policy()

        # Evaluation
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
                          random_seed=random_seed, agent_start_pos=agent_start_pos,
                          reward_fn=lambda grid, new_pos: goal_state_reward(grid, new_pos))

        Environment.evaluate_agent(grid, agent, iters, sigma,
                                   random_seed=random_seed,
                                   agent_start_pos=agent_start_pos)

if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed, args.agent, agent_start_pos=(3, 11))
