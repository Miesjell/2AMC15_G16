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
    from agents.monte_carlo_on_policy_agent import MonteCarloOnPolicyAgent
    #from world.environment import _distance_to_target_reward_function
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
    from agents.monte_carlo_on_policy_agent import MonteCarloOnPolicyAgent

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
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
    
    Args:
        agent_name: Name of the agent class to load (e.g., "RandomAgent").
        This should be the exact name of the class, not the module. 
        The class should be defined in a module named after the class in snake_case.
        For example, "RandomAgent" should be in a module named "random_agent.py".
    
    Returns:
        An instance of the specified agent class
    """
    try:
        # convert to snake_case for module name
        # Example: "RandomAgent" -> "random_agent"
        module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', agent_name).lower()
        
        # Import module
        module = importlib.import_module(f"agents.{module_name}")
        
        # Get the class and instantiate it
        agent_class = getattr(module, agent_name)
        print(f"Loaded agent class: {agent_class}")
        return agent_class()
    except (ImportError, AttributeError) as e:
        print(f"Error loading agent '{agent_name}': {e}")
        print("Falling back to RandomAgent.")
        return RandomAgent()
    
def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, agent_name: str):
    """Main loop of the program."""

    for grid in grid_paths:
        
        # Set up the environment
        env = Environment(grid, no_gui,sigma=sigma, target_fps=fps, 
                          random_seed=random_seed, agent_start_pos=[3,11]) # This is for the board A1
        # env = Environment(grid, no_gui,sigma=sigma, target_fps=fps, 
        #                   random_seed=random_seed, agent_start_pos=[3,3])
        
        # Initialize agent
        agent = load_agent(agent_name)
        print(f"Agent: {agent}")
        
        max_episode_length = 300  # Maximum length of an episode
        #successful_episodes_total = 0

        for episode_num in range(iters):
            state = env.reset()
            episode = []
            steps = 0
            episode_reward = 0
            reward_log = []
            #reached_goal = False

            # if successful_episodes_total >= 3:
            #     agent.epsilon = max(agent.epsilon * agent.decay, agent.min_epsilon)

            # if episode_num > 500:
            #     agent.epsilon = 0.05
            
            while steps < max_episode_length:
                action = agent.take_action(state)
                state, reward, terminated, info = env.step(action)
                episode_reward += reward
                episode.append((state, action, reward))
                steps += 1

                # if terminated and info.get("reached_target", False):
                #     successful_episodes_total += 1
                #     reached_goal = True
                #     break

                if terminated:
                    break
                
            agent.update(state, reward, action, episode)
            agent.decay_epsilon()

            # Check convergence
            if episode_num % 100 == 0 and agent.has_converged():
                print(f"[Converged] Q-values stable at episode {episode_num}")

            # Check positional behavior
            if episode_num % 100 == 0:
                agent.analyze_behavior(episode)

            # Periodic debugging
            if episode_num % 500 == 0:
                print(f"Episode {episode_num} | Epsilon: {agent.epsilon:.3f}")
                for s in list(agent.policy.keys())[:3]:
                    print(f"Policy at {s}: {agent.policy[s]}")

        Environment.evaluate_agent(grid, agent, iters, sigma,
                                   random_seed=random_seed, agent_start_pos=[3,11])
if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed, args.agent)