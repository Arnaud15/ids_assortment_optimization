import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pickle
from args import get_experiment_args

PAPER_EXPLORATION_BONUS = False # whether or not to employ the exploration bonus produced in the paper 
PAPER_UNDEFINED_PRIOR = True # wheter or not to employ the paper's faulty gaussian approximations
BAD_ITEM_CONSTANT = 0.5 # max possible preference for bad items
TOP_ITEM_CONSTANT = 1.
OUTPUTS_FOLDER = 'outputs'

if not os.path.isdir(OUTPUTS_FOLDER):
    os.makedirs(OUTPUTS_FOLDER)

AGENT_IDS = {'ts': "Thompson Sampling",
             'rd': "random",
             'ids': "Information Directed Sampling",
             'ets': "Epoch based Thompson Sampling",
             'etscs': "Epoch based TS with Correlated Sampling",
             'hts': 'Hypermodel TS',
             "eids": "Epoch based IDS",
             "evids": "Epoch based VIDS",
             'hids': "Hypermodel IDS"}

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
    noise_breaking_ties = np.random.randn(*belief.shape) * 1e-5
    belief += noise_breaking_ties
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
        return [prev_list + [i] for prev_list in prev_lists for i in range(prev_list[-1] + 1, n_items)]# TODO fix with greedy


def print_actions_posteriors(agent, past_observations):
    data_test = agent.sample_from_posterior(1000)
    print(f"agent posterior sample: {data_test.mean(0)}, {data_test.std(0)}")
    item_proposals = []
    for assortment, _ in past_observations:
        item_proposals += list(assortment)
    print(f"agent actions taken: {sorted([(key, i) for (key, i) in Counter(item_proposals).items()], key=lambda x:x[0])}")


def generate_hypersphere(dim, n_samples, norm=1):
    if norm==1:
        samples = np.random.rand(n_samples, dim)
        samples = samples / np.expand_dims(np.abs(samples).sum(1), 1)
        return samples
    elif norm==2:
        samples = np.random.randn(n_samples, dim)
        samples = samples / np.expand_dims(np.sqrt((samples ** 2).sum(1)), 1)
        return samples
    else:
        raise ValueError


def get_prior(n_items, prior_type="uniform"):
    if prior_type == "uniform":
        prior = np.random.rand(n_items + 1)
    elif prior_type == "restricted":
        prior = np.random.rand(n_items + 1)
        # Most items have preferences quite low (below 0.2)
        prior *= BAD_ITEM_CONSTANT 
        # First item is the best with maximum preferences
        prior[0] = 1. * TOP_ITEM_CONSTANT
    else:
        raise ValueError("Incorrect prior type, choice of 'uniform', 'restricted'")
    prior[-1] = 1.
    return prior


def print_regret(exp_names, exp_base_name):
    """
    :param exp_names: list of names for experiment data saved in the outputs folder
    :return: regret plots are saved in OUTPUTS_FOLDER
    """
    print(f"Regret plot for experiments of type: {exp_base_name}")

    plt.figure()

    for name in exp_names:
        exp_data = load_experiment_data(name)

        n_runs = len(exp_data)
        regrets = sum([run['best_reward'] - run['rewards'] for run in exp_data]) / n_runs
        n_steps = regrets.shape[0]
        cumulative_regret = np.cumsum(regrets)

        agent_name = AGENT_IDS[name.split('_')[0]]
        print(agent_name, n_runs, n_steps)
        curve_name = f"{agent_name} agent." if 'ids' not in name else f"{agent_name} agent, {name.split('_')[1]} samples, {name.split('_')[2]} action selection"
        plt.plot(np.arange(n_steps), cumulative_regret, label=curve_name)

    plt.xlabel('Time steps')
    plt.ylabel('Regret')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUTS_FOLDER, f'regret_{exp_base_name}.png'))
    plt.close()


if __name__ == "__main__":
    args = get_experiment_args(run_or_plot='plot')
    experiment_base_name = '_'.join([str(x) for x in [args.n, args.k, args.horizon, args.name]])
    experiments_to_plot = [agent_key + '_' + experiment_base_name for agent_key in args.agents]
    print_regret(experiments_to_plot, experiment_base_name)
