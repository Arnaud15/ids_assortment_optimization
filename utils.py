import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import minimize_scalar
from functools import partial
import pickle

# TODO refactor plots et prettify them

OUTPUTS_FOLDER = 'outputs'
if not os.path.isdir(OUTPUTS_FOLDER):
    os.makedirs(OUTPUTS_FOLDER)

AGENT_IDS = {'ts': "thompson_sampling",
             'rd': "random",
             'ids': "information_directed_sampling",
             'ats': "approximate_thompson_sampling"}


def save_experiment_data(exp_id, exp_data):
    path = os.path.join(OUTPUTS_FOLDER, exp_id + '.pickle')
    try:
        with open(path, 'rb') as handle:
            past_data = pickle.load(handle)
            exp_data += past_data
    except FileNotFoundError:
        pass
    with open(path, 'wb') as handle:
        pickle.dump(exp_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_experiment_data(name):
    path = os.path.join(OUTPUTS_FOLDER, name + '.pickle')
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def act_optimally(belief, top_k):
    if len(belief.shape) <= 1:
        return np.sort(np.argpartition(belief, -top_k)[-top_k:])
    else:
        return np.sort(np.argpartition(belief, -top_k, axis=1)[:, -top_k:], axis=1)


def possible_actions(n_items, assortment_size):
    assert assortment_size >= 1
    if assortment_size == 1:
        return [[i] for i in range(n_items)]
    else:
        prev_lists = possible_actions(n_items, assortment_size - 1)
        return [tuple(prev_list + [i]) for prev_list in prev_lists for i in range(prev_list[-1] + 1, n_items)]


def information_ratio_(rho, d1, d2, g1, g2):
    return (d1 * rho + (1 - rho) * d2) ** 2 / (g1 * rho + (1 - rho) * g2)


def optimized_ratio(d1, d2, g1, g2):
    func = partial(information_ratio_, d1=d1, d2=d2, g1=g1, g2=g2)
    solution = minimize_scalar(fun=func, bounds=(0., 1.), method='bounded')
    return solution.fun, solution.x


def expected_reward(preferences, action):
    """
    :param preferences: shape (m, n) model parameters
    :param action: indexes in [0, ..., n-1]
    :return:
    """
    filtered_preferences = preferences[:, action]
    return (filtered_preferences / (1 + filtered_preferences)).mean()


def print_regret(exp_names):
    """
    :param n_steps:
    :param assortment_size:
    :param true_preferences:
    :param exp_results: list of
    :return: plots and saves in OUTPUTS_FOLDER
    """

    plt.figure()
    for name in exp_names:
        exp_data = load_experiment_data(name)
        n_runs = len(exp_data)
        regrets = sum([run['best_reward'] - run['rewards'] for run in exp_data]) / n_runs
        n_steps = regrets.shape[0]
        cumulative_regret = np.cumsum(regrets)
        agent_name = AGENT_IDS[name.split('_')[0]]
        print(agent_name, n_runs, n_steps)
        plt.plot(np.arange(n_steps), cumulative_regret, label=f"Regret curve for {agent_name} agent.")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUTS_FOLDER, 'regret.png'))
    plt.close()


def print_actions(exp_results, true_preferences):
    plt.figure()
    preferences = true_preferences[:-1]
    preferences = preferences / preferences.sum()
    plt.scatter(np.arange(preferences.shape[0], dtype=int), preferences, label='preferences_normalized')
    for agent_name, (observations, _) in exp_results.items():
        item_proposals = defaultdict(int)

        for assortment, item_picked in observations:
            for item in assortment:
                item_proposals[item] += 1

        items, proposals = list(zip(*[(item, proposed_count) for item, proposed_count in item_proposals.items()]))
        proposals = np.array(proposals)
        proposals = proposals / proposals.sum()
        plt.scatter(items, proposals, label=f'proposals_normalized_probas for {agent_name} agent')

    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUTS_FOLDER, 'proposals_vs_preferences.png'))
    plt.close()


def print_run(env, agent, h, observations, rews):
    """
    :param env:
    :param agent:
    :param h:
    :param observations:
    :param rews:
    :return:
    """
    # Parameters
    k = agent.assortment_size
    n = agent.n_items
    true_preferences = env.preferences

    # Actions selected
    item_proposals = defaultdict(int)
    item_picks = defaultdict(int)

    for assortment, item_picked in observations:
        for item in assortment:
            item_proposals[item] += 1
        item_picks[item_picked] += 1

    # Cumulated rewards vs top agent expectation
    plt.figure()
    preferences_top = np.sort(true_preferences)[-(k + 1):]
    assert len(preferences_top) == k + 1
    preferences_top = preferences_top / preferences_top.sum()
    expected_top_rewards = preferences_top[:k].sum() * np.ones(h)
    expected_top_rewards = np.cumsum(expected_top_rewards)
    plt.plot(np.arange(h), expected_top_rewards, label="expected top cumulated rewards")
    plt.plot(np.arange(h), np.cumsum(rews), label="agent cumulated rewards")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUTS_FOLDER, 'rewards.png'))
    plt.close()

    plt.figure()
    items, proposals = list(zip(*[(item, proposed_count) for item, proposed_count in item_proposals.items()]))
    proposals = np.array(proposals) / h
    proposals = proposals / proposals.sum()
    plt.scatter(items, proposals, label='proposals_normalized_probas')
    preferences = true_preferences[:n]
    preferences = preferences / preferences.sum()
    plt.scatter(np.arange(n, dtype=int), preferences, label='preferences_normalized')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUTS_FOLDER, 'proposals_vs_preferences.png'))
    plt.close()

    plt.figure()
    items, picks = list(zip(*[(item, picked_count) for item, picked_count in item_picks.items()]))
    plt.scatter(items, np.array(picks) / h, label="actual_picks")
    true_preferences = env.preferences
    plt.scatter(np.arange(n + 1, dtype=int), true_preferences / true_preferences.sum(),
                label="picks_from_full_assortments")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUTS_FOLDER, 'picks_vs_full_assortment.png'))
    plt.close()

    return


if __name__ == "__main__":
    print_regret(['ids_5_2_80', 'ts_5_2_80', 'ats_5_2_80', 'rd_5_2_80'])
