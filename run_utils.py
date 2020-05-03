from typing import Tuple
from argparse import Namespace
import os
import numpy as np
from collections import Counter
import pickle
from args import OUTPUTS_FOLDER, BAD_ITEM_CONSTANT, TOP_ITEM_CONSTANT
from tqdm import tqdm


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


def summarize_run(observations, n_items):
    """
    param: observations = [obs=(k-sparse assortment given, index of item selected) for obs in observations]
    return: {assortments: 1D array of size K with how many times each item is proposed,
             picks: 1D array with how many times each item if picked}
    """
    run_assortments = sum(
        [assortment for assortment, item_picked in observations]
    )
    run_picks = {item_ix: 0 for item_ix in range(n_items + 1)}
    for assortment, item_picked in observations:
        run_picks[item_picked] += 1
    return {"assortments": run_assortments, "picks": run_picks}


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
