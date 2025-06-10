"""
Train your RL Agent in this file.
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
    from world import Environment
    from world.continuousEnvironment import ContinuousEnvironment

    from agents.random_agent import RandomAgent
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(
        path.join(path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.extend(root_path)
    from world import Environment
    from world.continuousEnvironment import ContinuousEnvironment

    from agents.random_agent import RandomAgent
    from agents.dqn_agent import DqnAgent


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument(
        "GRID",
        type=Path,
        nargs="+",
        help="Paths to the grid file to use. There can be more than one.",
    )
    p.add_argument(
        "--no_gui", action="store_true", help="Disables rendering to train faster"
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Sigma value for the stochasticity of the environment.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second to render at. Only used if no_gui is not set.",
    )
    p.add_argument(
        "--iter", type=int, default=1000, help="Number of iterations to go through."
    )
    p.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed value for the environment.",
    )
    p.add_argument(
        "-a",
        "--agent",
        type=str,
        default="RandomAgent",
        help="Name of the agent class to use (e.g., RandomAgent)",
    )
    p.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=1,
    )
    return p.parse_args()


def load_agent(agent_name: str, env):
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
        module_name = re.sub(r"(?<!^)(?=[A-Z])", "_", agent_name).lower()

        # Import module
        module = importlib.import_module(f"agents.{module_name}")

        # Get the class and instantiate it
        agent_class = getattr(module, agent_name)
        print(f"Loaded agent class: {agent_class}")
        # pass grid to dqn
        if agent_name == "DQNAgent":
            grid_shape = env.grid.shape  # (H, W)
            return agent_class(env, grid_shape=grid_shape)
        else:
            return agent_class(env)
        #return agent_class(env)
    except (ImportError, AttributeError) as e:
        print(f"Error loading agent '{agent_name}': {e}")
        print("Falling back to RandomAgent.")
        return RandomAgent()


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


        startPos = None

        # Setting proper start positions for each grid
        if grid.name == "A1_grid.npy":
            startPos = [3,11]
        elif grid.name == "Maze_Grid.npy":
            startPos = [1,5]
        elif grid.name == "Risk_Grid.npy":
            startPos = [2,8]
        elif grid.name == "Open_Field.npy":
            startPos = [2,7]
        elif grid.name == "Open_Field_2.npy":
            startPos = [2,7]
        

        # Set up the environment
        env = ContinuousEnvironment(
        #env = Environment(
            grid,
            no_gui,
            sigma=sigma,
            target_fps=fps,
            random_seed=random_seed,
            agent_start_pos = startPos,
        )

        episode_numbers = []
        episode_returns = []

        # Initialize agent
        agent = load_agent(agent_name, env)
        print(f"Agent: {agent}")

        if agent.__str__() == "MC_Agent":
            # Corrected and updated training loop: Run 'iters' full episodes
            for episode, _ in enumerate(trange(episodes, desc="Training episodes")):
                state = env.reset(agent_start_pos=startPos,)
                step_count = 0
                episode_data = []
                
                while True:
                    action = agent.take_action(state)
                    next_state, reward, terminated, info = env.step(action)
                    episode_data.append((state, info["actual_action"], reward))
                    step_count += 1
                    if terminated or step_count >= getattr(agent, "max_episode_len", 3000):
                    
                        #if info.get("target_reached", False):   # only successful eps
                        agent.update(episode_data)
                        break

                    state = next_state
                agent.epsilon = max(0.05, agent.epsilon * 0.995)

                # Evaluate the agent and append simple total reward and episode number to lists. Evaluate per x episodes!
                Environment.evaluate_agent(grid, agent, iters, sigma,
                                        random_seed=random_seed,
                                        agent_start_pos=startPos,)
                if episode % 50 == 0:
                    total_return = Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed, agent_start_pos=startPos)
                    episode_returns.append(total_return)
                    episode_numbers.append(episode + 1)

            # Plot learning curve
            plt.plot(episode_numbers, episode_returns, label="Episode Return")
            plt.xlabel("Episode")
            plt.ylabel("Simple Total Reward")
            plt.title("Learning Curve")
            plt.grid(True)
            # Save plot to file
            grid_dir = Path("learning_curves") / grid.stem
            grid_dir.mkdir(parents=True, exist_ok=True)
            
            # Add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(grid_dir / f"{grid.stem}_curve_{timestamp}.png")

            # Save data to CSV to make one learning curve with multiple agents later
            csv_path = Path("learning_curves") / f"{agent.__class__.__name__}_curve.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "return"])
                writer.writerows(zip(episode_numbers, episode_returns))

        # Training for DQN
        elif agent.__class__.__name__ == "DqnMlpAgent":
            print("Training DQN Agent...")

            # Warm up replay buffer with random actions
            print("Warming up replay buffer...")
            state = env.reset(agent_start_pos=startPos)
            for _ in range(1000):  # Increased warmup
                action = np.random.randint(0, agent.action_dim)
                next_state, reward, terminated, _ = env.step(action)
                agent.update(state, env.grid.copy(), reward, action, next_state, env.grid.copy(), terminated)
                if terminated:
                    state = env.reset(agent_start_pos=startPos)
                else:
                    state = next_state

            print(f"Replay buffer warmed up with {len(agent.buffer)} experiences")

            # Training loop
            for episode in trange(episodes, desc="Training episodes"):
                state = env.reset(agent_start_pos=startPos)
                episode_reward = 0
                steps_in_episode = 0

                for step in range(iters):
                    # Take action
                    action = agent.take_action(state, env.grid.copy())
                    next_state, reward, terminated, info = env.step(action)

                    # Store experience
                    agent.update(state, env.grid.copy(), reward, action,
                                 next_state, env.grid.copy(), terminated)

                    episode_reward += reward
                    steps_in_episode += 1
                    state = next_state

                    # Perform learning updates more frequently
                    if len(agent.buffer) >= agent.batch_size and step % 4 == 0:
                        loss = agent.update_batch(num_updates=1)

                    if terminated:
                        break

                # Decay epsilon after each episode
                agent.decay_epsilon()

                # Logging
                # if episode % 10 == 0:
                #     print(f"Episode {episode}: Reward={episode_reward:.2f}, "
                #           f"Steps={steps_in_episode}, Epsilon={agent.epsilon:.3f}")

                # Evaluation every 50 episodes
                if episode % 200 == 0:
                    print(f"Evaluating at episode {episode}...")
                    # Set epsilon to 0 for evaluation (pure exploitation)
                    old_epsilon = agent.epsilon
                    agent.epsilon = 0.0

                    total_return = ContinuousEnvironment.evaluate_agent(
                        grid, agent, iters, sigma,
                        random_seed=random_seed,
                        agent_start_pos=startPos
                    )

                    # Restore epsilon
                    agent.epsilon = old_epsilon

                    episode_returns.append(total_return)
                    episode_numbers.append(episode + 1)
                    print(f"Evaluation return: {total_return}")

            # Plot learning curve
            plt.figure(figsize=(10, 6))
            plt.plot(episode_numbers, episode_returns, label="Episode Return", marker='o')
            plt.xlabel("Episode")
            plt.ylabel("Total Return")
            plt.title(f"DQN Learning Curve - {grid.stem}")
            plt.legend()
            plt.grid(True)

            # Save plot
            grid_dir = Path("learning_curves") / grid.stem
            grid_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(grid_dir / f"{grid.stem}_DQN_curve_{timestamp}.png")
            plt.close()

            # Save data to CSV
            csv_path = Path("learning_curves") / f"{agent.__class__.__name__}_curve.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "return"])
                writer.writerows(zip(episode_numbers, episode_returns))


        else:

            # Always reset the environment to initial state
            state = env.reset()
            for episode, _ in enumerate(trange(episodes, desc="Training episodes")):
                state = env.reset()  # Always reset the environment to initial state
                if hasattr(agent, "reset"):
                    agent.reset()

                for _ in range(iters):
                    action = agent.take_action(state)
                    state, reward, terminated, info = env.step(action)
                    agent.update(state, reward, info["actual_action"])
                    if terminated:
                        break

                # Evaluate the agent and append simple total reward and episode number to lists. Evaluate per x episodes!
                if episode % 50 == 0:
                    total_return = Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed, agent_start_pos=startPos)
                    episode_returns.append(total_return)
                    episode_numbers.append(episode + 1)

                agent.prev_state = None
                agent.prev_action = None

            # Plot learning curve
            plt.plot(episode_numbers, episode_returns, label="Episode Return")
            plt.xlabel("Episode")
            plt.ylabel("Simple Total Reward")
            plt.title("Learning Curve")
            plt.grid(True)
            # Save plot to file
            grid_dir = Path("learning_curves") / grid.stem
            grid_dir.mkdir(parents=True, exist_ok=True)
            
            # Add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(grid_dir / f"{grid.stem}_curve_{timestamp}.png")
            
            # Save data to CSV to make one learning curve with multiple agents later
            csv_path = Path("learning_curves") / f"{agent.__class__.__name__}_curve.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "return"])
                writer.writerows(zip(episode_numbers, episode_returns))

if __name__ == '__main__':
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
