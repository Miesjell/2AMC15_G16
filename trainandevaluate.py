from argparse import ArgumentParser
import importlib
import pickle
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from world.continuousEnvironment import ContinuousEnvironment as Environment


def parse_args():
    parser = ArgumentParser(description="Train and evaluate a reinforcement learning agent.")
    parser.add_argument("GRID", type=Path, nargs="+", help="Path(s) to the grid file(s)")
    parser.add_argument("--no_gui", action="store_true", help="Disable GUI")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise level in actions")
    parser.add_argument("--fps", type=int, default=30, help="Target frames per second")
    parser.add_argument("--iter", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("-a", "--agent", type=str, default="RandomAgent", help="Agent class name")
    parser.add_argument("-e", "--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--evaluate", action="store_true", help="Only run evaluation, skip training")
    parser.add_argument("--show_images", action="store_true", help="Display path visualization during evaluation")
    parser.add_argument("--save_images", action="store_true", help="Save evaluation path visualization")
    return parser.parse_args()


def load_agent(agent_name: str, env):
    module_name = ''.join(['_' + c.lower() if c.isupper() else c for c in agent_name]).lstrip('_')
    module = importlib.import_module(f"agents.{module_name}")
    agent_class = getattr(module, agent_name)
    return agent_class(env)


def train_agent(grid_paths, agent_name, no_gui, sigma, fps, random_seed, iters, episodes):
    results_dir = Path("training_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = results_dir / f"{agent_name}_curve.csv"

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "return", "success"])

    for grid in grid_paths:
        start_pos = [8, 2] if grid.name == "mainrestaurant.npy" else [8, 2.2]
        env = Environment(grid_fp=grid, no_gui=no_gui, sigma=sigma, target_fps=fps,
                          random_seed=random_seed, agent_start_pos=start_pos)
        agent = load_agent(agent_name, env)
        print(f"Training agent: {agent_name} on grid: {grid.name}")

        for episode in range(episodes):
            state = env.reset(agent_start_pos=start_pos)
            total_return = 0.0
            success = False

            for _ in range(iters):
                action = agent.take_action(state)
                state, reward, done, info = env.step(action)
                agent.update(state, reward, info.get("actual_action", None))
                total_return += reward
                if done:
                    success = True
                    break

            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([episode + 1, total_return, int(success)])
            print(f"Episode {episode + 1}: Return={total_return:.2f}, Success={success}")

        # Save agent
        agent_save_path = results_dir / f"{agent_name}_final_agent.pkl"
        with open(agent_save_path, "wb") as f:
            pickle.dump(agent, f)
        print(f"Saved trained agent to {agent_save_path}")


def evaluate_agent(grid_paths, agent_name, sigma, iters, random_seed, show_images, save_images):
    results_dir = Path("evaluation_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = results_dir / f"{agent_name}_eval.csv"

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["grid", "return", "success", "steps"])

    agent_save_path = Path("learning_curves") / f"{agent_name}_final_agent.pkl"
    with open(agent_save_path, "rb") as f:
        agent = pickle.load(f)

    for grid in grid_paths:
        start_pos = [8, 2] if grid.name == "mainrestaurant.npy" else [8, 2.2]
        env = Environment(grid_fp=grid, no_gui=True, sigma=sigma, target_fps=-1,
                          random_seed=random_seed, agent_start_pos=start_pos)
        agent.env = env  # Ensure agent has correct environment
        print(f"Evaluating agent: {agent_name} on grid: {grid.name}")

        total, success, steps, img = Environment.evaluate_agent(
            grid_fp=grid,
            agent=agent,
            max_steps=iters,
            sigma=sigma,
            agent_start_pos=start_pos,
            random_seed=random_seed,
            show_images=show_images,
        )

        if save_images:
            img_path = results_dir / f"{agent_name}_{grid.stem}_eval.png"
            plt.imsave(img_path, img)

        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([grid.name, total, int(success), steps])

        print(f"[Eval] Grid: {grid.name}, Return={total:.2f}, Success={success}, Steps={steps}")


def main():
    args = parse_args()

    # Train the agent
    train_agent(
        grid_paths=args.GRID,
        agent_name=args.agent,
        no_gui=args.no_gui,
        sigma=args.sigma,
        fps=args.fps,
        random_seed=args.random_seed,
        iters=args.iter,
        episodes=args.episodes,
    )

    print("\nRunning evaluation of the trained agent...")

    # Evaluate the trained agent
    evaluate_agent(
        grid_paths=args.GRID,
        agent_name=args.agent,
        sigma=args.sigma,
        iters=args.iter,
        random_seed=args.random_seed,
        show_images=args.show_images,
        save_images=args.save_images,
    )


if __name__ == "__main__":
    main()
