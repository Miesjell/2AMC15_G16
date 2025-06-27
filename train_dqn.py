
"""
Train a DQN (or compatible) agent in a continuous environment and log results.
Supports multiple runs for statistical comparison and saves episode returns, success, and steps to CSV.
"""

from argparse import ArgumentParser
import importlib
from pathlib import Path
import csv
import numpy as np

from world.continuousEnvironment import ContinuousEnvironment as Environment



def parse_args():
    """
    Parse command-line arguments for training configuration.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = ArgumentParser()
    p.add_argument("grid", type=Path, help="Grid file path")
    p.add_argument("-a", "--agent", type=str, default="DQNAgent",
                   help="Name of the agent class to use (DQNAgent")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--agent_size", type=float, default=1.0)
    return p.parse_args()



def load_agent(agent_name: str, env):
    """
    Dynamically import and instantiate the agent class by name.
    Args:
        agent_name (str): Name of the agent class.
        env: Environment instance to pass to the agent.
    Returns:
        Instantiated agent object.
    """
    module_name = ''.join(['_' + c.lower() if c.isupper() else c for c in agent_name]).lstrip('_')
    module = importlib.import_module(f"agents.{module_name}")
    return getattr(module, agent_name)(env)



def train_agent(grid_path, agent_name, episodes, iters, sigma, fps, random_seed, agent_size, no_gui, num_runs=5):
    """
    Train the specified agent in the environment for multiple runs and episodes.
    Logs episode returns, success, and steps to CSV for each run.
    Args:
        grid_path: Path to grid configuration file.
        agent_name: Name of the agent class to use.
        episodes: Number of episodes per run.
        iters: Max steps per episode.
        sigma: Environment noise parameter.
        fps: Frames per second for environment.
        random_seed: Initial random seed (overridden for each run).
        agent_size: Size of the agent in the environment.
        no_gui: Disable GUI rendering if True.
        num_runs: Number of independent runs for statistics.
    """
    results_dir = Path("experiment-stepsize0.5ddaffy-dqn")
    results_dir.mkdir(exist_ok=True, parents=True)
    start_pos = [8, 2]  # Fixed agent start position

    for run in range(num_runs):
        random_seed = np.random.randint(0, 1_000_000)  # Use a new random seed for each run
        print(f"\n=== Starting run {run+1}/{num_runs} with random_seed={random_seed} ===")

        # Create environment and agent
        env = Environment(
            grid_fp=grid_path,
            no_gui=no_gui,
            sigma=sigma,
            target_fps=fps,
            random_seed=random_seed,
            agent_start_pos=start_pos,
            agent_size=agent_size,
        )
        agent = load_agent(agent_name, env)

        # Prepare output CSV for this run
        csv_file = results_dir / f"{agent_name}_curve_run{run+1}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "return", "success", "steps"])

        # Training loop for each episode
        for ep in range(episodes):
            state = env.reset(agent_start_pos=start_pos)
            if hasattr(agent, "reset"):
                agent.reset()

            total_return = 0
            steps = 0
            success = False

            for _ in range(iters):
                action = agent.take_action(state)
                state, reward, done, info = env.step(action)
                agent.update(state, reward, info.get("actual_action", None))
                total_return += reward
                steps += 1


                if done:
                    success = True
                    break

            # Log episode results
            with open(csv_file, "a", newline="") as f:
                csv.writer(f).writerow([ep + 1, total_return, int(success), steps])

            print(f"[Train] [Run {run+1}] Ep {ep + 1}: Return={total_return:.2f}, Success={success}, Steps={steps}")

        eval_episodes = 10  # Number of evaluation episodes
        eval_returns = []
        for eval_ep in range(eval_episodes):
            eval_return = Environment.evaluate_agent(
                grid_fp=args.grid,
                agent=agent,
                max_steps=iters,
                sigma=sigma,
                agent_start_pos=start_pos,
                random_seed=random_seed + eval_ep, 
                show_images=False,
            )
            eval_returns.append(eval_return)
            print(f"[Eval] Episode {eval_ep+1}: Return={eval_return:.2f}")

        eval_dir = results_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / f"{agent_name}_eval_curve.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["eval_episode", "return"])
            for i, ret in enumerate(eval_returns):
                writer.writerow([i+1, ret])

        print(f"[Eval] Average Return over {eval_episodes} episodes: {np.mean(eval_returns):.2f}")

if __name__ == "__main__":
    args = parse_args()
    train_agent(
        grid_path=args.grid,
        agent_name=args.agent,
        episodes=args.episodes,
        iters=args.iters,
        sigma=args.sigma,
        fps=args.fps,
        random_seed=args.random_seed,
        no_gui=args.no_gui,
        agent_size=args.agent_size,
    )