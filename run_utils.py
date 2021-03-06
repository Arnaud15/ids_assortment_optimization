import os
import pickle
import time
from argparse import Namespace
from tqdm import tqdm
from typing import Tuple, List
from collections import defaultdict
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from env import AssortmentEnvironment, BernoulliSemi, CombEnv
from base_agents import BayesAgent
from args import (
    BAD_ITEM_CONSTANT,
    TOP_ITEM_CONSTANT,
    RAW_OUTPUTS_FOLDER,
    AGG_OUTPUTS_FOLDER,
)


def params_to_gaussian(posterior):
    # gaussian_stds = np.array(
    #     [np.sqrt(b_ / a_ * ((b_ / a_) + 1) / a_) for (a_, b_) in posterior],
    # )
    # gaussian_means = np.array([b_ / a_ for (a_, b_) in posterior],)
    gaussian_means = posterior[1] / posterior[0]
    gaussian_stds = np.sqrt(
        gaussian_means * (gaussian_means + 1.0) / posterior[0]
    )
    return gaussian_means, gaussian_stds


def args_to_exp_id(args: Namespace,) -> str:
    """
    :param args: experiment parameters NameSpace
    :return: String to identify the experiment parameters
    """
    identifier = f"N{args.N}"
    identifier += f"_K{args.K}"
    identifier += f"_T{args.T}"
    identifier += f"_prior{args.prior.upper()}"
    if args.prior == "full_sparse":
        identifier += f"_fallbackP{int(args.p * 100)}"
    return identifier


def args_to_agent_name(args: Namespace,) -> Tuple[str, str]:
    """
    Returns the agent_key and agent_name from run parameters.
    """
    # Parsing agent class
    if args.prior == "full_sparse":
        agent_key = args.agent
        agent_name = args.agent.upper()
    else:
        agent_key = "e" + args.agent
        agent_name = "Epoch" + args.agent.upper()
    if agent_name[-2:] == "RD":
        pass
    elif agent_name[-2:] == "TS":
        sampling_names = {0: "default", 1: "correlated", 2: "optimistic"}
        agent_name += sampling_names[args.sampling]
        if args.prior == "full_sparse":
            if args.optim_prob is None:
                optim_p = "default"
            else:
                optim_p = f"{int(100 * args.optim_prob)}"
            agent_name += optim_p
    else:
        assert agent_name[-3:] == "IDS"
        if "CIDS" not in agent_name:
            print("Not ECIDS")
            if args.info_type == "variance":
                agent_name = agent_name[:-3] + "V" + agent_name[-3:]
            agent_name += args.objective
            if args.objective == "lambda":
                agent_name += args.scaling
            agent_name += f"M{args.M}"
            agent_name += f"dyn{args.dynamics}"
        else:
            agent_name += args.info_type
    return agent_key, agent_name


def run_episode(
    envnmt: CombEnv, actor: BayesAgent, n_steps: int,
) -> Tuple[np.ndarray, dict]:
    """
    :param n_steps: simulation horizon
    :return:
    rewards = 1D numpy array of size (horizon,)
    with entries = expected_reward from action taken given env parameters
    """
    # Initialization of observations and agent
    envnmt.reset()
    actor.reset()
    rewards = np.zeros(n_steps)
    for ix in tqdm(range(n_steps)):
        # act / step / update
        assortment = actor.act()
        obs, reward = envnmt.step(assortment)
        actor.update_posterior(obs)
        # Store expected reward, observation
        rewards[ix] = reward
    print(f"selected: {[envnmt.selections[i] for i in range(actor.n_items)]}")
    print(f"proposed: {[envnmt.counts[i] for i in range(actor.n_items)]}")
    return rewards, actor.stored_info()


def save_experiment_data(exp_id: str, exp_data: List[dict], target: str):
    """
    :param exp_id: save identifier
    :param exp_data: List of dictionaries (as many as nruns in the experiment)
    each dictionary is:
    {
    rewards:numpy.ndarray of rewards in run,
    best_reward:expected_reward_from_opt_action_run,
    logs:logs from the agent
    }
    """
    if target == "raw_folder":
        path = os.path.join(RAW_OUTPUTS_FOLDER, exp_id)
        path += f"_{int(time.time() * 1e5)}.pickle"
    elif target == "agg_folder":
        path = os.path.join(AGG_OUTPUTS_FOLDER, exp_id)
    else:
        raise ValueError("Choice of ['raw_folder', 'agg_folder'].")
    with open(path, "wb") as handle:
        pickle.dump(exp_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def filter_saved_files(exp_id, target):
    assert target in {"raw_folder", "agg_folder"}, "Incorrect target."
    retained_filenames = set()
    it = (
        os.listdir(RAW_OUTPUTS_FOLDER)
        if target == "raw_folder"
        else os.listdir(AGG_OUTPUTS_FOLDER)
    )
    for filename in it:
        if filename.startswith(exp_id):
            if target == "raw_folder":
                t_filename = filename[: -(len(filename.split("_")[-1]) + 1)]
                retained_filenames.add(t_filename)
            else:
                retained_filenames.add(filename)
                print(filename)
    return [fname for fname in retained_filenames]


def aggregate(file_root):
    agg_data = defaultdict(list)
    count_runs = 0
    count_files = 0
    T = 0
    for filename in os.listdir(RAW_OUTPUTS_FOLDER):
        if filename.startswith(file_root):
            count_files += 1
            exp_data = load_experiment_data(filename, target="raw_folder")
            for run in exp_data:
                count_runs += 1
                steps_data = run["logs"]["steps"]
                best_r = run["best_reward"]
                agg_data["rewards"].append(
                    [best_r - r for r in run["rewards"]]
                )
                agg_data["steps"].append(steps_data)
                if not T:
                    T = len(run["rewards"])
                if ("EpochIDS" in file_root) or ("EpochVIDS" in file_root):
                    assert len(
                        run["logs"]["info_ratio"]
                    ), "No information ratio logs."
                    info_ratios = np.zeros(T)
                    rhos = np.zeros(T)
                    entropies = np.zeros(T)
                    delta_mins_2 = np.zeros(T)
                    ix = -1
                    for index, step in enumerate(steps_data):
                        if step:
                            ix += 1
                        info_ratios[index] = run["logs"]["info_ratio"][ix]
                        rhos[index] = run["logs"]["rho_policy"][ix]
                        entropies[index] = run["logs"]["entropy_a_star"][ix]
                        delta_mins_2[index] = run["logs"]["delta_min_2"][ix]
                    agg_data["info_ratios"].append(info_ratios)
                    agg_data["entropies"].append(entropies)
                    agg_data["delta_mins_2"].append(delta_mins_2)
    for data_key, data_lists in agg_data.items():
        agg_data[data_key] = np.vstack(data_lists)
        assert agg_data[data_key].shape == (count_runs, T)
        if data_key in {"rewards", "steps"}:
            agg_data[data_key] = agg_data[data_key].cumsum(1)
        agg_data[data_key] = (
            agg_data[data_key].mean(0),
            agg_data[data_key].std(0),
            agg_data[data_key].shape[0],
        )
    print(
        f"Total of {count_files} files saved, {count_runs} runs saved for {file_root}."
    )
    with open(os.path.join(AGG_OUTPUTS_FOLDER, file_root), "wb") as handle:
        pickle.dump(agg_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Data saved.")


def plot_results(file_root, key_to_plot):
    exp_title, truncated_label = (
        file_root.split("_")[:-1],
        file_root.split("_")[-1],
    )
    exp_title = " ".join(exp_title)
    data_loaded = load_experiment_data(file_root, target="agg_folder")
    if key_to_plot in data_loaded:
        y_data = data_loaded[key_to_plot][0]
        x_data = np.arange(1, y_data.shape[0] + 1)
        errors = (
            1.96
            * data_loaded[key_to_plot][1]
            / np.sqrt(data_loaded[key_to_plot][2])
        )
        plt.plot(x_data, y_data, label=truncated_label)
        plt.fill_between(
            x_data, y_data + errors, y_data - errors, alpha=0.1, color="red"
        )
        plt.title(exp_title)
        # if key_to_plot == "info_ratios":
        #     plt.ylim(0., 10.0)
    else:
        print(f"Info: {key_to_plot} not in {file_root}.")


def load_experiment_data(name, target):
    if target == "raw_folder":
        # Loading raw data form many agents
        path = os.path.join(RAW_OUTPUTS_FOLDER, name)
    elif target == "agg_folder":
        path = os.path.join(AGG_OUTPUTS_FOLDER, name)
    else:
        raise ValueError("Choice of ['raw_folder', 'agg_folder'].")
    with open(path, "rb") as handle:
        return pickle.load(handle)


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
        prior = np.random.rand(n_items)
    elif prior_type == "full_sparse":
        top_item = np.random.randint(1, n_items)
        prior = np.zeros(n_items)
        prior[top_item] = np.inf
        prior[0] = fallback_weight
    else:
        raise ValueError("Choice of 'uniform', 'full_sparse'")
    return prior
