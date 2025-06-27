
"""
Analyze and visualize agent size experiment results.
Loads experiment CSVs, aggregates statistics, and plots learning curves, success rates, and average steps.
"""

import pandas as pd
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_episode_data(exp_dirs):
    """
    Combine all experiment results from the given directories into a single DataFrame.
    Each row represents an episode from a specific run, with agent type and size inferred from directory name.
    Args:
        exp_dirs (list of str or Path): List of experiment result directories.
    Returns:
        pd.DataFrame: Combined episode data from all runs and experiments.
    Raises:
        FileNotFoundError: If a directory does not exist.
        ValueError: If directory naming does not match expected pattern or run id cannot be parsed.
        RuntimeError: If no CSV files are found.
    """
    records = []

    for exp_dir in map(Path, exp_dirs):
        if not exp_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {exp_dir}")
        # Infer agent size and type from directory name
        parts = exp_dir.name.split("-")
        try:
            agent_size = float(parts[1].replace("agentsize", "")) 
            agent = "DQN" if "dqn" in parts[2].lower() else "PPO"
        except (IndexError, ValueError):
            raise ValueError(
                f"Directory naming must follow pattern 'experiment-agentsize<val>-<algo>', got: {exp_dir.name}"
            )

        # Load each run's CSV and annotate with metadata
        for csv_file in exp_dir.glob("*_curve_run*.csv"):
            run_id_match = re.search(r"run(\d+)", csv_file.stem)
            if not run_id_match:
                raise ValueError(f"Cannot parse run-id from: {csv_file.name}")
            run_id = int(run_id_match.group(1))

            df = pd.read_csv(csv_file)
            df["agent"]  = agent
            df["agent_size"]  = agent_size
            df["run_id"] = run_id
            records.append(df)
    
    if not records:
        raise RuntimeError("No CSV files found in the provided directories.")

    episodes = pd.concat(records, ignore_index=True)
    return episodes


def plot_learning_curves_from_df(
    episodes,
    group_vars=("agent", "agent_size"),
    window=50,
    save_dir=Path("results_2")
):
    """
    Plot learning curves (return, success, steps) with mean ±95% CI, averaged over runs for each experimental setting.
    Args:
        episodes (pd.DataFrame): Combined episode data for all runs and experiments.
        group_vars (tuple): Columns to group by (e.g., agent, agent_size).
        window (int): Rolling window size for smoothing.
        save_dir (Path): Directory to save plots.
    """
    sns.set(style="whitegrid")
    metrics = {
        "return":  ("Total Return",  "compare_learning_curves"),
        "success": ("Success Rate",  "compare_success_rates"),
        "steps":   ("Average Steps", "compare_avg_steps"),
    }

    # Sort episodes df to keep rolling windows in chronological order
    episodes = episodes.sort_values(["run_id", "episode"]).copy()

    for col, (ylabel, fname_prefix) in metrics.items():
        plt.figure(figsize=(10, 5))

        # Compute rolling average for each run
        episodes[f"{col}_roll"] = (
            episodes
            .groupby(list(group_vars) + ["run_id"])[col]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )

        # Aggregate the runs, using mean and Standard Error of the Mean
        summary = (
            episodes
            .groupby(list(group_vars) + ["episode"])[f"{col}_roll"]
            .agg(mean="mean",
                 sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)))
            .reset_index()
        )

        # Plot the average curve per experimental group, with 95% CI ribbon
        for name, sub in summary.groupby(list(group_vars)):
            label = ", ".join(f"{k}={v}" for k, v in zip(group_vars, name)) \
                    if isinstance(name, tuple) else str(name)
            plt.plot(sub["episode"], sub["mean"], label=label)
            plt.fill_between(
                sub["episode"],
                sub["mean"] - 1.96 * sub["sem"],
                sub["mean"] + 1.96 * sub["sem"],
                alpha=0.25,
            )

        plt.title(f"{ylabel} (Rolling Avg, window={window})")
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_dir.mkdir(exist_ok=True, parents=True)
        fname = save_dir / f"{fname_prefix}_{'_'.join(group_vars)}.png"
        plt.savefig(fname, dpi=300)
        plt.show()

# Construct dataframe with all runs of the experiment
episodes = load_episode_data(['experiment-agentsize0.5-dqn', 'experiment-agentsize0.5-ppo', 'experiment-agentsize1.5-dqn', 'experiment-agentsize1.5-ppo'])

# Compute success rate per run and the mean of the total return in the final episodes
grouped = (episodes
           .groupby(["agent", "agent_size", "run_id"])
           .agg(
               success_rate_last50 = ("success", lambda s: s.tail(50).mean()),
               mean_totalreturn_last50 = ("return",  lambda s: s.tail(50).mean()),
               std_totalreturn_last50 = ("return",  lambda s: s.tail(50).std(ddof=1))
           )
           .reset_index())
output_dir = Path("results_2")
output_dir.mkdir(exist_ok=True)
grouped.to_csv(output_dir / "summary_step_size_experiment.csv", index=False)
# Get mean for these values for all experiment combinations
summary = (
    grouped
    .groupby(["agent", "agent_size"])
    .agg(
        avg_success_rate = ("success_rate_last50", "mean"),
        avg_total_return_mean = ("mean_totalreturn_last50", "mean"),
        avg_total_return_std = ("std_totalreturn_last50", "mean") 
    )
    .reset_index()
)
print(summary)

# Plot learning and succes rate curves
plot_learning_curves_from_df(
    episodes,
    group_vars=("agent", "agent_size"),
    window=50,
    save_dir=Path("results_2")
)
