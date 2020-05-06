from env import AssortmentEnvironment
from base_agents import RandomAgent
from ts_agents import EpochSamplingTS, SparseTS
from ids_agents import EpochSamplingIDS, SparseIDS
from run_utils import (
    args_to_exp_id,
    proba_to_weight,
    get_prior,
    run_episode,
    save_experiment_data,
    summarize_run,
)
import numpy as np
from args import get_experiment_args


# LIST OF SUPPORTED AGENT KEYS WITH THE CORRESPONDING CLASSES
AGENTS = {
    "rd": RandomAgent,
    "ets": EpochSamplingTS,
    "eids": EpochSamplingIDS,
    "ts": SparseTS,
    "ids": SparseIDS,
}

# TODO: refactor the scaler search below
def scaler_search(args, agent_class, info_type):
    """
    args = Namespace object with the parameters for the experiment
    agent_class = class for the IDS agent for which we find the best epsilon parameter for greedy IDS action selection
    """
    print("Scale search begins:")
    # Grid search over the following possible values
    scales = np.linspace(start=0.0, stop=0.05, num=20)
    best_rewards = -np.inf
    for scale in scales:
        # Instantiate the agent
        agent = agent_class(
            k=args.k,
            n=args.n,
            correlated_sampling=False,
            horizon=args.horizon,
            n_samples=args.ids_samples,
            limited_prefs=True if args.prior == "restricted" else False,
            info_type=info_type,
            action_type=args.ids_action_selection,
            scaling_factor=scale,
            params=args,
        )
        # Examine the sum of rewards accumulated on average over args.best_scaler_n runs
        sum_of_rewards = 0.0
        for _ in range(args.best_scaler_n):
            run_preferences = get_prior(n_items=args.n, prior_type=args.prior)
            env = AssortmentEnvironment(n=args.n, v=run_preferences)
            top_preferences = np.sort(run_preferences)[-(args.k + 1) :]
            top_preferences = top_preferences / top_preferences.sum()
            expected_reward_from_best_action = top_preferences[: args.k].sum()

            obs_run, rewards_run = run_episode(
                envnmt=env, actor=agent, n_steps=args.best_scaler_h
            )
            sum_of_rewards += (
                rewards_run - expected_reward_from_best_action
            ).sum()
        sum_of_rewards = sum_of_rewards / args.best_scaler_n
        if sum_of_rewards > best_rewards:
            best_rewards = sum_of_rewards
            best_scale = scale
            print(
                f"New best scaling factor:{scale} with average cumulative regret: {sum_of_rewards:.2f} over horizon: {args.best_scaler_h}"
            )
    return best_scale


if __name__ == "__main__":
    # Getting arguments
    args = get_experiment_args(run_or_plot="run")
    print(f"Arguments are {args}")

    # Printing environment type = prior distribution for environment parameters
    print(f"Environment type is: {args.prior}.")
    if args.agent[0] == 'e':
        assert(args.prior in {"uniform", "soft_sparse"})
    elif args.agent != "rd":
        assert(args.prior == "full_sparse")
    # TODO arg for env type: three possibilities = uniform, soft_sparse, full_sparse

    # Parsing agent name and parameters
    correlated_sampling = bool(args.correlated_sampling)
    fallback_item_weight = proba_to_weight(args.p)

    # Agent name (string used in plots to identify it compared to other agents)
    _, exp_id = args_to_exp_id(
        agent_name=args.agent, args=args, plotting=False
    )
    print(f"Exp identifier is {exp_id}")

    # Picking the correct agent class
    agent_class = AGENTS[args.agent]

    # Looking for the best approximation to true IDS action selection here
    if (
        (args.agent == "eids")
        and (args.ids_action_selection == "greedy")
        and args.find_best_scaler
    ):
        args.greedy_scaler = scaler_search(args, agent_class, args.info_type)

    # Instantiating the agent before experiments
    agent = agent_class(
        k=args.k,
        n=args.n,
        correlated_sampling=args.correlated_sampling,
        fallback_weight=fallback_item_weight,
        fallback_proba=args.p,
        horizon=args.horizon,
        limited_prefs=True if args.prior == "soft_sparse" else False,
        n_samples=args.ids_samples,
        info_type=args.info_type,
        action_type=args.ids_action_selection,
        scaling_factor=args.greedy_scaler,
        params=args,
    )

    # Actual experiments with printing
    experiment_data = []
    for _ in range(args.nruns):
        run_preferences = get_prior(
            n_items=args.n,
            prior_type=args.prior,
            fallback_weight=fallback_item_weight,
        )
        env = AssortmentEnvironment(n=args.n, v=run_preferences)
        if args.prior == "full_sparse":
            expected_reward_from_best_action = 1.0
        else:
            top_preferences = np.sort(run_preferences)[-(args.k + 1) :]
            top_preferences = top_preferences / top_preferences.sum()
            expected_reward_from_best_action = top_preferences[: args.k].sum()

        run_data = {"best_reward": expected_reward_from_best_action}
        obs_run, rewards_run = run_episode(
            envnmt=env, actor=agent, n_steps=args.horizon, verbose=args.verbose
        )

        run_data["rewards"] = rewards_run
        run_data.update(summarize_run(observations=obs_run, n_items=args.n))

        experiment_data.append(run_data)

    save_experiment_data(exp_id, experiment_data)
    print(f"Experiment successfully terminated")

else:
    import sys

    sys.exit("main should be used as a script")
