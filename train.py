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
                   help="Number of episodes to run during training.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("-a", "--agent", type=str, default="RandomAgent",
                   help="Name of the agent class to use (e.g., RandomAgent)")
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

def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, agent_name: str, agent_start_pos=(3, 11)):

    for grid in grid_paths:
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
                          random_seed=random_seed, agent_start_pos=agent_start_pos)
        agent = load_agent(agent_name)
        print(f"Agent: {agent}")

        # Corrected and updated training loop: Run 'iters' full episodes
        for episode in trange(iters, desc="Training Episodes"):
            state = env.reset(agent_start_pos=agent_start_pos)
            step_count = 0
            episode = []
            
            while True:
                action = agent.take_action(state)
                next_state, reward, terminated, info = env.step(action)
                episode.append((state, info["actual_action"], reward))
                step_count += 1
                if terminated or step_count >= getattr(agent, "max_episode_len", 3000):
                
                    #if info.get("target_reached", False):   # only successful eps
                    agent.update(episode)
                    break

                state = next_state
            agent.epsilon = max(0.05, agent.epsilon * 0.995)


        

        # Evaluate the agent
        Environment.evaluate_agent(grid, agent, iters, sigma,
                                   random_seed=random_seed,
                                   agent_start_pos=agent_start_pos)

if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed, args.agent, agent_start_pos=(3, 11))
