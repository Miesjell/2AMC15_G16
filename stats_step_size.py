import pandas as pd
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data from experiments
def load_episode_data(exp_dirs):
    """ Function to combine all experiment results into one df
    """
    records = []

    for exp_dir in map(Path, exp_dirs):
        if not exp_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {exp_dir}")
        
        # Add column with agent name and experiment value for each episode, inferred from file name
        parts = exp_dir.name.split("-")
        try:
            step_size = float(parts[1].replace("step_size", "")) 
            agent = "DQN" if "dqn" in parts[2].lower() else "PPO"
        except (IndexError, ValueError):
            raise ValueError(f"Directory naming must follow pattern "
                             f"'experiment-step_size<val>-<algo>', got: {exp_dir.name}")

        # Load each file, add column run id
        for csv_file in exp_dir.glob("*_curve_run*.csv"):
            run_id_match = re.search(r"run(\d+)", csv_file.stem)
            if not run_id_match:
                raise ValueError(f"Cannot parse run-id from: {csv_file.name}")
            run_id = int(run_id_match.group(1))

            df = pd.read_csv(csv_file)
            df["agent"]  = agent
            df["step_size"]  = step_size
            df["run_id"] = run_id
            records.append(df)
    
    if not records:
        raise RuntimeError("No CSV files found in the provided directories.")

    episodes = pd.concat(records, ignore_index=True)
    return episodes

# Function to plot graphs learning curves, success rates and avg steps
def plot_learning_curves_from_df(
    episodes,
    group_vars=("agent", "steps"),
    window=50,
    save_dir=Path("results_2")
):
    """
    Draw learning curves (return, success, steps) with mean ±95 % CI,
    averaged over the runs for each experimental setting.
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

        # Aggregate the 10 runs, using mean and Standard Error of the Mean
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
episodes = load_episode_data(['experiment-step_size0.5-dqn', 'experiment-step_size0.5-ppo', 'experiment-step_size1.0-dqn', 'experiment-step_size1.0-ppo'])

# Compute success rate per run and the mean of the total return in the final episodes
grouped = (episodes
           .groupby(["agent", "step_size", "run_id"])
           .agg(
               success_rate_last50 = ("success", lambda s: s.tail(50).mean()),
               mean_totalreturn_last50 = ("return",  lambda s: s.tail(50).mean()),
               std_totalreturn_last50 = ("return",  lambda s: s.tail(50).std(ddof=1))
           )
           .reset_index())
output_dir = Path("results_2")
output_dir.mkdir(exist_ok=True)
grouped.to_csv(output_dir / "summary_step_size_experiment.csv", index=False)

# Plot learning and succes rate curves
plot_learning_curves_from_df(
    episodes,
    group_vars=("agent", "step_size"),
    window=50,
    save_dir=Path("results_2")
)
