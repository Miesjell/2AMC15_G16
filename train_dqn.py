from argparse import ArgumentParser
import importlib
from pathlib import Path
import csv
import pickle
import numpy as np

from world.continuousEnvironment import ContinuousEnvironment as Environment
from agents.p_p_o_agent import PPOAgent


def parse_args():
    p = ArgumentParser()
    p.add_argument("grid", type=Path, help="Grid file path")
    p.add_argument("-a", "--agent", type=str, default="RandomAgent")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--random_seed", type=int, default=0)
    return p.parse_args()


def load_agent(agent_name: str, env):
    module_name = ''.join(['_' + c.lower() if c.isupper() else c for c in agent_name]).lstrip('_')
    module = importlib.import_module(f"agents.{module_name}")
    return getattr(module, agent_name)(env)


def train_agent(grid_path, agent_name, episodes, iters, sigma, fps, random_seed, no_gui):
    results_dir = Path("learning_curves")
    results_dir.mkdir(exist_ok=True, parents=True)
    start_pos = [8, 2] if grid_path.name == "mainrestaurant.npy" else [8, 2.2]

    env = Environment(
        grid_fp=grid_path,
        no_gui=no_gui,
        sigma=sigma,
        target_fps=fps,
        random_seed=random_seed,
        agent_start_pos=start_pos,
    )

    agent = load_agent(agent_name, env)

    # Output CSV
    csv_file = results_dir / f"{agent_name}_curve.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "success", "steps"])

    # Training loop
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

            if isinstance(agent, PPOAgent) and info.get("target_reached", False):
                agent.goal_reached_once = True
                agent.entropy_coef = 0.0
                agent.buffer = []

            if done:
                success = True
                break

        with open(csv_file, "a", newline="") as f:
            csv.writer(f).writerow([ep + 1, total_return, int(success), steps])

        print(f"[Train] Ep {ep + 1}: Return={total_return:.2f}, Success={success}, Steps={steps}")


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
    )
