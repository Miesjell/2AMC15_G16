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

    plt.figure(figsize=(10, 5))

    # Plot learning curves over a window of 50 episodes
    for agent, df in learning_curves.items():
        df["return_rolling"] = df["return"].rolling(window=50, min_periods=1).mean()
        plt.plot(df["episode"], df["return_rolling"], label=f"{agent} Rolling Avg (50)")

    plt.title("Learning Curves for All Agents")
    plt.xlabel("Episode")
    plt.ylabel("Total Return (Rolling Avg, window=50)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(results_dir / f"compare_learning_curves_{timestamp}.png")
    plt.show()

    # Plot success rate (moving average or every 50th episode)
    plt.figure(figsize=(10, 5))
    for agent, df in learning_curves.items():
        # Calculate rolling success rate (window=50)
        if "success" in df.columns:
            df["success_rate"] = df["success"].rolling(window=50, min_periods=1).mean()
            plt.plot(df["episode"], df["success_rate"], label=f"{agent} Success Rate")
        else:
            print(f"No 'success' column found for {agent}")

    plt.title("Success Rate (Moving Average, window=50)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f"compare_success_rates_{timestamp}.png")
    plt.show()

    # Plot average steps per episode
    plt.figure(figsize=(10, 5))
    for agent, df in learning_curves.items():
        if "steps" in df.columns:
            df["avg_steps"] = df["steps"].rolling(window=50, min_periods=1).mean()
            plt.plot(df["episode"], df["avg_steps"], label=f"{agent} Avg Steps")
        else:
            print(f"No 'steps' column found for {agent}")
    
    plt.title("Average Steps per Episode (Moving Average, window=50)")
    plt.xlabel("Episode")
    plt.ylabel("Average Steps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(results_dir / f"compare_avg_steps_{timestamp}.png") 
    plt.show()

    for agent, df in learning_curves.items():
        # Last batch indices
        last_batch = df.tail(50)
        
        # Print last batch average return
        if "return" in df.columns:
            avg_return_last = last_batch["return"].mean()
            print(f"{agent} average return (last 50 episodes): {avg_return_last:.2f}")
        
        # Print last batch success rate
        if "success" in df.columns:
            success_rate_last = last_batch["success"].mean()
            print(f"{agent} success rate (last 50 episodes): {success_rate_last:.2%}")
        
        # Print last batch average steps
        if "steps" in df.columns:
            avg_steps_last = last_batch["steps"].mean()
            print(f"{agent} average steps (last 50 episodes): {avg_steps_last:.2f}")

# Example usage
agents = ["PPOAgent"]
results_dir = Path("learning_curves")
plot_learning_curves(agents, results_dir)