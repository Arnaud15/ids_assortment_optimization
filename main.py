import sys
import os
from env import AssortmentEnvironment
from base_agents import RandomAgent
from ts_agents import EpochSamplingTS
from ids_agents import EpochSamplingIDS, EpochSamplingCIDS
from run_utils import (
    args_to_exp_id,
    args_to_agent_name,
    proba_to_weight,
    get_prior,
    run_episode,
    save_experiment_data,
    aggregate,
    filter_saved_files,
    plot_results,
)
from args import get_experiment_args, PLOTS_FOLDER
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm


# LIST OF AGENT KEYS WITH THE CORRESPONDING CLASSES
AGENTS = {
    "rd": RandomAgent,
    "erd": RandomAgent,
    "ets": EpochSamplingTS,
    "eids": EpochSamplingIDS,
    "ecids": EpochSamplingCIDS,
}


def run_from_args(run_args, exp_name):
    """
    Run experiments for a given agent,
    for a given number of runs,
    a given env type.
    """
    assert run_args.agent is not None, "Agent not specified for 'run' script."
    logging.basicConfig(
        level=logging.DEBUG,
        filename="logs.log",
        format="%(levelname)s:%(message)s",
    )

    agent_key, agent_name = args_to_agent_name(run_args)
    print(f"Agent ID is: {agent_name}.")
    agent = AGENTS[agent_key](
        k=run_args.K,
        n=run_args.N,
        sampling=run_args.sampling,
        n_samples=run_args.M,
        info_type=run_args.info_type,
        params=run_args,
    )

    # Actual experiments with printing
    experiment_data = []
    for _ in tqdm(range(run_args.nruns)):
        run_preferences = get_prior(
            n_items=run_args.N,
            prior_type=run_args.prior,
            fallback_weight=proba_to_weight(run_args.p),
        )
        env = AssortmentEnvironment(item_prefs=run_preferences)
        run_data = {"best_reward": env.r_star_from_subset_size(run_args.K)}
        rewards_run, actor_logs = run_episode(
            envnmt=env, actor=agent, n_steps=run_args.T,
        )
        run_data["rewards"] = rewards_run
        run_data["logs"] = actor_logs

        experiment_data.append(run_data)

    saving_path = exp_name + "_" + agent_name
    save_experiment_data(saving_path, experiment_data, target="raw_folder")
    print(f"Experiment successfully terminated.")
    print(f"Saved with name: {saving_path}.")


def agg_results(exp_args, exp_name):
    """
    Aggregate experimental results,
    for several agents,
    fixed environment params.
    """
    file_roots = filter_saved_files(exp_name, target="raw_folder")
    for file_root in file_roots:
        print(file_root)
        aggregate(file_root)


def plot_runs(exp_args, exp_name):
    """
    Plot experimental results,
    for several agents,
    with aggregated results,
    for fixed environment params.
    """
    file_roots = filter_saved_files(exp_name, target="agg_folder")
    labels_retained = ["rewards", "info_ratios", "entropies", "delta_mins_2"]
    for lab in labels_retained:
        plt.figure()
        for file_root in file_roots:
            plot_results(file_root, key_to_plot=lab)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(PLOTS_FOLDER, exp_name) + "_" + lab + ".png")
        plt.close()


if __name__ == "__main__":
    # Getting arguments
    args = get_experiment_args()

    # Script type
    print(f"Script type is {args.script}.")

    # Experiment Name
    exp_name = args_to_exp_id(args)
    print(f"Exp identifier:{exp_name}.")
    # Env params
    # print(
    #     f"""Environment:
    # N={args.N}, K={args.K}, T={args.T},
    # prior={args.prior}.
    # """
    # )

    if args.script == "run":
        run_from_args(args, exp_name)
    elif args.script == "agg":
        agg_results(args, exp_name)
    elif args.script == "plot":
        plot_runs(args, exp_name)
    else:
        sys.exit("Incorrect script argument.")
else:
    sys.exit("main should be used as a script")
