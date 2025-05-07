"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
import importlib
from pathlib import Path
import re
from tqdm import trange

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import json  # Assuming Q-tables are saved as JSON files
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
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--iter", type=int, default=10000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("-a", "--agent", type=str, default="RandomAgent",
                   help="Name of the agent class to use (e.g., RandomAgent)")
    return p.parse_args()


def load_agent(agent_name: str, env: Environment):
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
        return agent_class(env)
    except (ImportError, AttributeError) as e:
        print(f"Error loading agent '{agent_name}': {e}")
        print("Falling back to RandomAgent.")
        return RandomAgent()

def animate_q_tables(folder_path: str ="q_tables"):
    """
    Create an animated heatmap from Q-tables stored as JSON files in a folder.

    Args:
        folder_path (str): Path to the folder containing Q-table files.
    """
    # First ensure numpy is installed
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Required libraries not found. Install them with:")
        print("pip install numpy matplotlib seaborn")
        return None
        
    # Load Q-tables from files
    q_tables = []
    for file_name in sorted(os.listdir(folder_path)):  # Sort to ensure correct order
        if file_name.endswith(".json"):  # Assuming Q-tables are saved as JSON
            with open(os.path.join(folder_path, file_name), "r") as f:
                q_table_dict = json.load(f)
                q_tables.append(q_table_dict)

    # Create the animation
    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame):
        ax.clear()
        q_table = q_tables[frame]
        
        # Extract states and coordinates
        states = []
        for state_str in q_table.keys():
            # Parse string representation of tuple like "(1, 1)"
            coords = state_str.strip('()').split(',')
            x = int(coords[0].strip())
            y = int(coords[1].strip())
            states.append((x, y))
        
        # Find maximum coordinate values to determine grid size
        max_x = max([s[0] for s in states]) + 1
        max_y = max([s[1] for s in states]) + 1
        
        # Initialize the grid for max Q-values
        grid = np.zeros((max_x, max_y))
        # Mark states that don't have data with NaN (will show as a different color)
        grid.fill(np.nan)
        
        # Populate grid with maximum Q-values for each state
        for state_str, actions in q_table.items():
            coords = state_str.strip('()').split(',')
            x = int(coords[0].strip())
            y = int(coords[1].strip())
            
            if actions:  # Check if actions dictionary is not empty
                # Store only maximum Q-value for this state
                grid[x, y] = max(actions.values())
        
        # Create heatmap - use a masked array to handle NaN values
        masked_grid = np.ma.masked_invalid(grid)
        cmap = plt.cm.coolwarm
        cmap.set_bad('lightgray')  # Color for NaN values
        
        sns.heatmap(
            masked_grid,
            annot=True,  # Show the actual max Q-values
            fmt=".2f",   # Format with 2 decimal places
            cmap=cmap,
            cbar_kws={'label': 'Max Q-Value'},
            ax=ax,
        )
        
        ax.set_title(f"Max Q-values at Episode {frame + 1}")
        ax.set_xlabel("Y coordinate")
        ax.set_ylabel("X coordinate")
        ax.invert_yaxis()  # Make the origin bottom-left like a standard coordinate system

    ani = FuncAnimation(fig, update, frames=len(q_tables), repeat=False)
    
    # Save the animation
    ani.save("q_table_animation.mp4", writer="ffmpeg", fps=10)
    
    return ani

    
def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, agent_name: str):
    """Main loop of the program."""

    for grid in grid_paths:
        
        # Set up the environment
        env = Environment(grid, no_gui,sigma=sigma, target_fps=fps, 
                          random_seed=random_seed, agent_start_pos=(1,1))
        
        # Initialize agent
        agent = load_agent(agent_name, env)
        print(f"Agent: {agent}")
        
        for episode in range(1000):
            print(f"Episode: {episode}")
            # Always reset the environment to initial state
            state = env.reset()
            for _ in trange(iters):
                
                # Agent takes an action based on the latest observation and info.
                action = agent.take_action(state)

                state, reward, terminated, info = env.step(action)
                
                # If the final state is reached, stop.
                if terminated:
                    break
        # Log the Q-table every 10 episodes
                agent.update(state, reward, info["actual_action"])

            
            # Evaluate the agent
            Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)
        # if (episode + 1) % 10 == 0:
            #     print(f"Q-Table after episode {episode + 1}:")
            #     for state, actions in agent.q_table.items():
            #         print(f"State {state}: {actions}")
                
                # Save Q-table as a JSON file
            with open(f"q_tables/episode_{episode + 1}.json", "w") as f:
                json.dump({str(k): v for k, v in agent.q_table.items()}, f)

        animate_q_tables()    # The action is performed in the environment




if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed, args.agent)
