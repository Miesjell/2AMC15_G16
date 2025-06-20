from argparse import ArgumentParser
import importlib
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import trange
from datetime import datetime
import csv
import numpy as np
import pickle

from world.continuousEnvironment import ContinuousEnvironment as Environment
from agents.random_agent import RandomAgent

def load_agent(agent_name: str, env):
    module_name = ''.join(['_' + c.lower() if c.isupper() else c for c in agent_name]).lstrip('_')
    module = importlib.import_module(f"agents.{module_name}")
    agent_class = getattr(module, agent_name)
    return agent_class(env)

def parse_args():
    p = ArgumentParser(description="RL Trainer")
    p.add_argument("GRID", type=Path, nargs="+", help="Paths to the grid file(s)")
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--iter", type=int, default=1000)
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("-a", "--agent", type=str, default="RandomAgent")
    p.add_argument("-e", "--episodes", type=int, default=100)
    p.add_argument("--evaluate", action="store_true", help="Run in evaluation mode only")
    p.add_argument("--show_images", action="store_true", help="Display path visualization")
    p.add_argument("--save_images", action="store_true", help="Save path visualization images")
    return p.parse_args()

def run_evaluation(grid_paths, agent_name, sigma, iters, random_seed, episodes, show_images, save_images):
    results_dir = Path("evaluation_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = results_dir / f"{agent_name}_eval.csv"

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "return", "success", "steps"])

    for grid in grid_paths:

        start_pos = [8, 2.2]
        if grid.name == "mainrestaurant.npy":
            start_pos = [8, 2]

        env = Environment(
            grid_fp=grid,
            no_gui=True,
            sigma=sigma,
            target_fps=-1,
            random_seed=random_seed,
            agent_start_pos=start_pos,
        )

        agent_save_path = Path("learning_curves") / f"{agent_name}_final_agent.pkl"
        with open(agent_save_path, "rb") as f:
            agent = pickle.load(f)
        agent.env = env  # Ensure the agent has the correct environment
        print(f"Loaded agent: {agent_name} from {agent_save_path}")
  
        from world.continuousEnvironment import ContinuousEnvironment
        for episode in range(episodes):
            agent = load_agent(agent_name, env)  
            total, success, steps, img = ContinuousEnvironment.evaluate_agent(
                grid_fp=grid,
                agent=agent,
                max_steps=iters,
                sigma=sigma,
                agent_start_pos=start_pos,
                random_seed=random_seed,
                show_images=show_images,
            )

            if save_images:
                img_path = results_dir / f"{agent_name}_episode_{episode+1}.png"
                plt.imsave(img_path, img)

            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([episode + 1, total, int(success), steps])

            print(f"[Eval] Episode {episode + 1}: Return = {total:.2f}, Success = {success}, Steps = {steps}")

def main(grid_paths, no_gui, iters, fps, sigma, random_seed, agent_name, episodes, evaluate, show_images, save_images):
    if evaluate:
        run_evaluation(grid_paths, agent_name, sigma, iters, random_seed, episodes, show_images, save_images)
        return
    
    results_dir = Path("learning_curves")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = results_dir / f"{agent_name}_curve.csv"

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "return", "success"])

    #success_count = 0

    for grid in grid_paths:

        start_pos = [8, 2.2]
        if grid.name == "mainrestaurant.npy":
            start_pos = [8, 2]

        env = Environment(
            grid_fp=grid,
            no_gui=no_gui,
            sigma=sigma,
            target_fps=fps,
            random_seed=random_seed,
            agent_start_pos=start_pos,
        )
        agent = load_agent(agent_name, env)
        print(f"Loaded agent: {agent_name}")

        for episode in range(episodes):
            state = env.reset(agent_start_pos=start_pos)
            total_return = 0
            # Adding here a boolean for succes rate
            success = False
            for _ in range(iters):
                action = agent.take_action(state)
                state, reward, done, info = env.step(action)
                agent.update(state, reward, info.get("actual_action", None))
                total_return += reward
                if done:
                    success = True
                    break

            #if success:
            #    success_count += 1    

            # Log results to CSV
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([episode + 1, total_return, int(success)])

            print(f"Episode {episode + 1}: Total Return = {total_return} and Success = {success}")

    agent_save_path = results_dir / f"{agent_name}_final_agent.pkl"
    with open(agent_save_path, "wb") as f:
        pickle.dump(agent, f)
    print(f"Agent saved to {agent_save_path}")

# Run eveluating of the trained agend

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
        args.evaluate,
        args.show_images,
        args.save_images,
    )
