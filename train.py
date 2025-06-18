
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import csv
import os
from tqdm import trange

from world.continuousEnvironment import ContinuousEnvironment as Environment
from agents.stable_baselines3_agent import StableBaselines3Agent


def parse_args():
    p = ArgumentParser(description="RL Trainer with Stable-Baselines3")
    p.add_argument("GRID", type=Path, nargs="+", help="Paths to the grid file(s)")
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--iter", type=int, default=1000)
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("-a", "--algorithm", type=str, default="DQN")
    p.add_argument("-e", "--episodes", type=int, default=100)
    p.add_argument("--eval_freq", type=int, default=100)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--load_model", type=str, default=None)
    return p.parse_args()


def train_agent_episodic(agent, env, episodes, max_steps, eval_freq=100):
    episode_numbers = []
    episode_returns = []
    successful_episodes = 0

    print("🚀 Starting training with better exploration...")

    # Pre-fill the replay buffer
    print("📦 Pre-filling replay buffer with random steps...")
    agent.model.learn(total_timesteps=5000, reset_num_timesteps=True)

    for episode in trange(episodes):
        obs = agent.wrapped_env.reset()
        obs = obs[0]
        if hasattr(agent, 'reset'):
            agent.reset()

        episode_reward = 0
        steps = 0
        goal_reached = False

        for step in range(max_steps):
            action = agent.take_action(obs)
            next_obs, reward, done, info = agent.wrapped_env.step([action])
            next_obs = next_obs[0]
            reward = reward[0]
            done = done[0]

            if reward == 100.0:
                goal_reached = True
                successful_episodes += 1
                print(f"🎯 TARGET REACHED! Episode {episode}, Step {step} (Success #{successful_episodes})")

            episode_reward += reward
            obs = next_obs
            steps += 1

            if done:
                break

        # Start training earlier and only every 5 episodes
        # if episode % 2 == 0:
        agent.model.learn(total_timesteps=200, reset_num_timesteps=False)

        if episode % 100 == 0:
            success_rate = successful_episodes / max(1, episode + 1) * 100
            print(f"Episode {episode}: Reward={episode_reward:.2f}, Steps={steps}, Success Rate={success_rate:.1f}%")

        if episode % eval_freq == 0:
            total_return = Environment.evaluate_agent(
                grid_fp=env.grid_fp,
                agent=agent,
                max_steps=max_steps,
                sigma=env.sigma,
                agent_start_pos=env.agent_start_pos,
                show_images=False,
                verbose=False
            )
            print(f"Eval Episode {episode}: Return={total_return:.2f}")
            episode_numbers.append(episode + 1)
            episode_returns.append(total_return)

    final_success_rate = successful_episodes / episodes * 100
    print(f"Training completed. Final success rate: {final_success_rate:.1f}% ({successful_episodes}/{episodes})")

    return episode_numbers, episode_returns



def main():
    args = parse_args()
    os.makedirs("models", exist_ok=True)
    os.makedirs("learning_curves", exist_ok=True)

    for grid in args.GRID:
        print(f"Training on grid: {grid}")
        start_pos = [8, 2.2] if grid.name != "mainrestaurant.npy" else [8, 2]

        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from agents.stable_baselines3_agent import StableBaselines3Agent, GymnasiumWrapper

        base_env = Environment(
            grid_fp=grid,
            no_gui=args.no_gui,
            sigma=args.sigma,
            target_fps=args.fps,
            random_seed=args.random_seed,
            agent_start_pos=start_pos,
            agent_size = 0.5
        )

        wrapped_env = DummyVecEnv([lambda: GymnasiumWrapper(base_env)])
        env = VecNormalize(wrapped_env, norm_obs=True, norm_reward=False)

        agent = StableBaselines3Agent(env, args.algorithm)

        if args.load_model:
            print(f"Loading model from {args.load_model}")
            agent.load(args.load_model)

        print(f"Training {args.algorithm} for {args.episodes} episodes...")
        episode_numbers, episode_returns = train_agent_episodic(
            agent, base_env, args.episodes, args.iter, args.eval_freq
        )

        plt.figure()
        plt.plot(episode_numbers, episode_returns, label="Episode Return")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title(f"{args.algorithm} on {grid.stem}")
        plt.grid(True)

        curve_dir = Path("learning_curves") / grid.stem
        curve_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(curve_dir / f"{grid.stem}_{timestamp}.png")

        with open(curve_dir / f"{args.algorithm}_curve.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "return"])
            writer.writerows(zip(episode_numbers, episode_returns))

        plt.close()

        if args.save_model:
            model_path = f"models/{args.algorithm}_{grid.stem}_{timestamp}"
            agent.save(model_path)
            print(f"Model saved to {model_path}")

        print(f"Training completed. Results saved to {curve_dir}")


if __name__ == "__main__":
    main()
