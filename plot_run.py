from args import get_experiment_args, OUTPUTS_FOLDER
from run_utils import load_experiment_data, args_to_exp_id
import numpy as np
import matplotlib.pyplot as plt
import os


AGENT_NAMES = {
    "ts": ["Thompson Sampling", "blue"],
    "tscs": ["Thompson Sampling w/ Correlated Sampling", "purple"],
    "rd": ["Random", "green"],
    "idsgain": ["Information Directed Sampling", "red"],
    "idsvariance": ["Variance-based IDS", "orange"],
    "ets": ["Epoch based Thompson Sampling", "blue"],
    "etscs": ["Epoch based TS w/ Correlated Sampling", "purple"],
    "eidsgain": ["Epoch based IDS", "red"],
    "eidsvariance": ["Epoch based VIDS", "orange"],
}


def print_regret(exp_names, exp_base_name):
    """
    :param exp_names: exp_ids saved in the outputs folder
    :return: regret plots are saved in OUTPUTS_FOLDER
    """
    print(f"Regret plot for experiments of type: {exp_base_name}")

    plt.figure()

    for name in exp_names:
        exp_data = load_experiment_data(name)

        n_runs = len(exp_data)
        # regrets = (
        #     sum([run["best_reward"] - run["rewards"] for run in exp_data])
        #     / n_runs
        # )
        regrets = np.array(
            [run["best_reward"] - run["rewards"] for run in exp_data]
        )
        n_steps = regrets.shape[1]
        cumulative_regret = np.cumsum(regrets, axis=1)
        err = (
            cumulative_regret.std(0)
            * 1.96
            / np.sqrt(cumulative_regret.shape[0])
        )
        cumulative_regret = cumulative_regret.mean(0)

        agent_name, color = AGENT_NAMES[name.split("_")[0]]
        print(agent_name, n_runs, n_steps)
        curve_name = f"{agent_name} agent."
        plt.plot(
            np.arange(n_steps),
            cumulative_regret,
            label=curve_name,
            color=color,
        )
        plt.fill_between(
            np.arange(n_steps),
            cumulative_regret + err,
            cumulative_regret - err,
            color=color,
            alpha=0.1,
        )
    plt.xlabel("Time steps")
    plt.ylabel("Regret")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUTS_FOLDER, f"regret_{exp_base_name}.png"))
    plt.close()


if __name__ == "__main__":
    args = get_experiment_args(run_or_plot="plot")

    exp_base_name, _ = args_to_exp_id(
        agent_name=None, args=args, plotting=True
    )
    experiments_to_plot = [
        args_to_exp_id(agent_name=agent_key, args=args, plotting=True)[1]
        for agent_key in args.agents
    ]
    print(experiments_to_plot)
    print_regret(experiments_to_plot, exp_base_name)

