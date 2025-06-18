import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def plot_learning_curves(agents, results_dir):
    learning_curves = {}

    # Load learning curves from CSV files
    for agent in agents:
        csv_file = results_dir / f"{agent}_curve.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            learning_curves[agent] = df
        else:
            print(f"CSV not found for {agent}: {csv_file}")

    # Plot learning curves
    for agent, df in learning_curves.items():
        plt.plot(df["episode"], df["return"], label=agent)

    plt.title("Learning Curves for All Agents")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.legend()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(results_dir / f"compare_learning_curves_{timestamp}.png")
    plt.show()

# Example usage
agents = ["DQNAgent", "RandomAgent", "NullAgent"]
results_dir = Path("learning_curves")
plot_learning_curves(agents, results_dir)


# # Define training settings
# agents = ["DQNagent", "RandomAgent", "NullAgent"]
# grid_path = "grid_configs/mainrestaurant.npy"
# episodes = 1000
# sigma = 0
# results_dir = Path("learning_curves")

# # Ensure results directory exists
# results_dir.mkdir(parents=True, exist_ok=True)

# # Full path to Python in the current venv
# python_exec = sys.executable

# # Store loaded curves
# learning_curves = {}

# # Run each agent and collect its CSV
# for agent in agents:
#     print(f"Running {agent}...")
#     csv_file = results_dir / f"{agent}_curve.csv"

#     # Run the training script
#     subprocess.run([
#         python_exec, "train.py",
#         grid_path,
#         "--agent", agent,
#         "--episodes", str(episodes),
#         "--no_gui",
#         "--sigma", str(sigma)
#     ], check=True)

#     # Load the learning curve CSV written by train.py
#     if csv_file.exists():
#         df = pd.read_csv(csv_file)
#         learning_curves[agent] = df
#     else:
#         print(f"CSV not found for {agent}: {csv_file}")

# # Plot the learning curves
# for agent, df in learning_curves.items():
#     plt.plot(df["episode"], df["return"], label=agent)

# plt.title("Learning Curves for All Agents")
# plt.xlabel("Episode")
# plt.ylabel("Simple Total Reward")
# plt.grid(True)
# plt.legend()

# # Save figure
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# plt.savefig(results_dir / f"compare_learning_curves_{timestamp}.png")