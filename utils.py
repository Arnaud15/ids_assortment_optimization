from typing import Tuple
from argparse import Namespace
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pickle
from args import get_experiment_args
from tqdm import tqdm

# EPOCH BASED SAMPLING PARAMS
# Whether to employ the exploration bonus introduced in the paper
PAPER_EXPLORATION_BONUS = False
# Whether to employ the paper's faulty gaussian approximations
PAPER_UNDEFINED_PRIOR = True
BETA_RVS = True

# SOFT_SPARSE SETTING PARAMS
BAD_ITEM_CONSTANT = 0.5  # soft_sparse preference for bad items
TOP_ITEM_CONSTANT = 1.0  # preference for (know) top item in soft-sparse
OUTPUTS_FOLDER = "outputs"
if not os.path.isdir(OUTPUTS_FOLDER):
    os.makedirs(OUTPUTS_FOLDER)

AGENT_NAMES = {
    "ts": "Thompson Sampling",
    "tscs": "Thompson Sampling w/ Correlated Sampling",
    "rd": "Random",
    "idsgain": "Information Directed Sampling",
    "idsvariance": "Variance-based IDS",
    "ets": "Epoch based Thompson Sampling",
    "etscs": "Epoch based TS w/ Correlated Sampling",
    "eidsgain": "Epoch based IDS",
    "eidsvariance": "Epoch based VIDS",
}


def args_to_exp_id(
    agent_name: str, args: Namespace, plotting: bool
) -> Tuple[str, str]:
    """
        :param args: experiment parameters NameSpace
        :param information_ratio_type: choice of 'gain' or 'variance'
        :return:
        (base_exp_id = env_params,
         agent_id = agent_and_env_parameters)
    """
    exp_keys = [
        args.n,
        args.k,
        int(args.p * 100),
        args.horizon,
        args.prior,
        args.name,
    ]
    base_exp_id = "_".join([str(elt) for elt in exp_keys])

    if agent_name is None:
        return base_exp_id, "whatever"
    else:
        agent_id = agent_name
        if "ids" in agent_id and (not plotting):
            agent_id += args.info_type
        elif (
            args.correlated_sampling and ("ts" in agent_id) and (not plotting)
        ):
            agent_id += "cs"
        if "ids" in agent_id:
            agent_id += f"_{args.ids_samples}_{args.ids_action_selection}"
    agent_id = agent_id + "_" + base_exp_id
    return base_exp_id, agent_id


def run_episode(envnmt, actor, n_steps, verbose=False):
    """
    :param envnmt: instance from the AssortmentEnvironment class
    :param actor: instance from the Agent class
    :param n_steps: horizon of the run in the environment
    :return: (observations history = list of (assortment one-hot array of size N+1, 0<=index<=N of item picked),
    rewards = 1D numpy array of size (horizon,) with entries = expected_reward from action taken given env parameters
    """
    # Initialization of observations and agent
    envnmt.reset()
    actor.reset()
    top_item = envnmt.top_item
    rewards = np.zeros(n_steps)
    obs = [0] * n_steps
    iterator = tqdm(range(n_steps)) if not verbose else range(n_steps)
    for ix in iterator:
        # act / step / update
        assortment = actor.act()
        item_selected = envnmt.step(assortment)
        actor.update(item_selected)
        # Store expected reward, observation
        obs[ix] = (assortment, item_selected)
        if top_item is not None and top_item in assortment:
            rewards[ix] = 1.0
        else:
            unnorm_pick_proba = envnmt.preferences[assortment].sum()
            rewards[ix] = unnorm_pick_proba / (1.0 + unnorm_pick_proba)
        # Print current posterior belief of the agent if asked
        if verbose and (
            (ix > n_steps - 2) or ((ix + 1) % 50 == 0) or (ix == 0)
        ):
            print_actions_posteriors(
                agent=actor, past_observations=obs[: ix + 1]
            )
    prefs_str = [
        f"{run_preference:.2f}" for run_preference in envnmt.preferences
    ]

    # Print environment model parameters if asked
    if verbose:
        print(f"Initial preferences were :{prefs_str}")
        print(
            f"Best action was: {envnmt.preferences.argsort()[-(actor.assortment_size + 1):][::-1][1:]}"
        )
    return obs, rewards


def save_experiment_data(exp_id, exp_data):
    """
    :param exp_id: name of the experient data
    :param exp_data: list of dictionaries (as many as nruns in the experiment)
    each dictionary is:
    {
    rewards:numpy.ndarray of rewards in run,
    best_reward:expected_reward_from_opt_action_run,
    assortments: {item: how_many_times_proposed for item in nitems} for run
    picks: {item:how_many_times_picked for item in nitems+1} for run
    }
    """
    path = os.path.join(OUTPUTS_FOLDER, exp_id + ".pickle")
    try:
        with open(path, "rb") as handle:
            past_data = pickle.load(handle)
            exp_data += past_data
    except FileNotFoundError:
        pass
    with open(path, "wb") as handle:
        pickle.dump(exp_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_experiment_data(name):
    path = os.path.join(OUTPUTS_FOLDER, name + ".pickle")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def act_optimally(belief, top_k):
    noise_breaking_ties = np.random.randn(*belief.shape) * 1e-5
    belief += noise_breaking_ties
    if len(belief.shape) <= 1:
        return np.sort(np.argpartition(belief, -top_k)[-top_k:])
    else:
        return np.sort(
            np.argpartition(belief, -top_k, axis=1)[:, -top_k:], axis=1
        )


def possible_actions(n_items, assortment_size):
    assert assortment_size >= 1
    if assortment_size == 1:
        return [[i] for i in range(n_items)]
    else:
        prev_lists = possible_actions(n_items, assortment_size - 1)
        return [
            prev_list + [i]
            for prev_list in prev_lists
            for i in range(prev_list[-1] + 1, n_items)
        ]  # TODO fix with greedy


def print_actions_posteriors(agent, past_observations):
    data_test = agent.sample_from_posterior(1000)
    print(f"agent posterior sample: {data_test.mean(0)}, {data_test.std(0)}")
    item_proposals = []
    for assortment, _ in past_observations:
        item_proposals += list(assortment)
    print(
        f"agent actions taken: {sorted([(key, i) for (key, i) in Counter(item_proposals).items()], key=lambda x:x[0])}"
    )


def proba_to_weight(p0: float) -> float:
    assert p0 <= 1.0 and p0 >= 0.0
    # p0 = w / (1 + w) => w = p0 / (1 - p0)
    return p0 / (1 - p0)


def get_prior(
    n_items: int, prior_type: str, fallback_weight: float
) -> np.ndarray:
    """
        :param n_items:int: N parameter = number of items 
        :param fallback_weight = weight of the fallback_item
        :param prior_type: choice of (uniform, soft_sparse, full_sparse) 
    """
    if prior_type == "uniform":
        prior = np.random.rand(n_items + 1)
    elif prior_type == "soft_sparse":
        prior = np.random.rand(n_items + 1)
        # Most items have preferences quite low (below 0.2)
        prior *= BAD_ITEM_CONSTANT
        # First item is the best with maximum preferences
        prior[0] = 1.0 * TOP_ITEM_CONSTANT
    elif prior_type == "full_sparse":
        top_item = np.random.randint(1, n_items)
        prior = np.zeros(n_items + 1)
        prior[top_item] = np.inf
        prior[0] = fallback_weight
    else:
        raise ValueError("Choice of 'uniform', 'soft_sparse', 'full_sparse'")
    prior[-1] = 1.0
    return prior


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
        regrets = (
            sum([run["best_reward"] - run["rewards"] for run in exp_data])
            / n_runs
        )
        n_steps = regrets.shape[0]
        cumulative_regret = np.cumsum(regrets)

        agent_name = AGENT_NAMES[name.split("_")[0]]
        print(agent_name, n_runs, n_steps)
        curve_name = (
            f"{agent_name} agent."
        )
        plt.plot(np.arange(n_steps), cumulative_regret, label=curve_name)

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
