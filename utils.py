import os
import numpy as np
import numba
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import minimize_scalar
from functools import partial
import pickle

DISCRETIZATION_IDS = 25
RHO_VALUES = np.linspace(start=0., stop=1.0, num=DISCRETIZATION_IDS)

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
        return [tuple(prev_list + [i]) for prev_list in prev_lists for i in range(prev_list[-1] + 1, n_items)]


def information_ratio_(rho, d1, d2, g1, g2):
    return (d1 * rho + (1 - rho) * d2) ** 2 / (g1 * rho + (1 - rho) * g2)


def optimized_ratio(d1, d2, g1, g2, discrete=True):
    func = partial(information_ratio_, d1=d1, d2=d2, g1=g1, g2=g2)
    possible_values = [(rho, func(rho=rho)) for rho in RHO_VALUES]
    min_ = np.inf
    rho_min = -1
    for (rho_, val) in possible_values:
        if val < min_:
            rho_min = rho_
            min_ = val
    return min_, rho_min
    # solution = minimize_scalar(fun=func, bounds=(0., 1.), method='bounded')
    # return solution.fun, solution.x


def expected_reward(preferences, action):
    """
    :param preferences: shape (m, n) sampled model parameters
    :param action: indexes in [0, ..., n-1]
    :return:
    """
    filtered_item_weights = preferences[:, action].sum(1)
    return (filtered_item_weights / (1 + filtered_item_weights)).mean()


def print_actions_posteriors(agent, past_observations):
    data_test = agent.sample_from_posterior(1000)
    print(f"agent posterior sample: {data_test.mean(0)}, {data_test.std(0)}")
    item_proposals = []
    for assortment, _ in past_observations:
        item_proposals += list(assortment)
    print(f"agent actions taken: {sorted([(key, i) for (key, i) in Counter(item_proposals).items()], key=lambda x:x[0])}")


def print_regret(exp_names):
    """
    :param exp_names: list of names for experiment data saved in the outputs folder
    :return: regret plots are saved in OUTPUTS_FOLDER
    """
    first_exp_name = exp_names[0]
    exp_base_name = first_exp_name.split('_')[2:] if 'ids' in first_exp_name else first_exp_name.split('_')[2:]
    exp_base_name = '_'.join(exp_base_name)
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
        curve_name = f"Regret curve for {agent_name} agent." if 'ids' not in name else f"Regret curve for {agent_name} agent with {name.split('_')[1]} samples"
        plt.plot(np.arange(n_steps), cumulative_regret, label=curve_name)

    plt.xlabel('Time steps')
    plt.ylabel('Regret')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUTS_FOLDER, f'regret_{exp_base_name}.png'))
    plt.close()


def print_actions(exp_names):
    """
    :param exp_names: list of names for experiment data saved in the outputs folder
    :return: action plots are saved in OUTPUTS_FOLDER
    """
    exp_base_name = exp_names[0].split('_')[1:]
    exp_base_name = '_'.join(exp_base_name)
    print(f"Action plot for experiments of type: {exp_base_name}")

    plt.figure()
    for name in exp_names:
        exp_data = load_experiment_data(name)

        assortments = sum([data_run['assortments'] for data_run in exp_data])
        n_items = len(assortments)

        picks = Counter()
        for data_run in exp_data:
            picks = picks + Counter(data_run['picks'])
        picks_sum = sum([npicks for npicks in picks.values()])

        agent_name = AGENT_IDS[name.split('_')[0]]
        print(agent_name, n_items)
        plt.scatter(np.arange(n_items + 1), [picks[item_id] / picks_sum for item_id in range(n_items)],
                    label=f"Normalized picks for {agent_name} agent.")
        plt.savefig(os.path.join(OUTPUTS_FOLDER, f'picks_{name}.png'))

        plt.scatter(np.arange(n_items), assortments / assortments.sum(),
                    label=f"Normalized assortments proposed by {agent_name} agent.")
        plt.savefig(os.path.join(OUTPUTS_FOLDER, f'assortments_{name}.png'))

    plt.legend()
    plt.grid()
    plt.close()

def generate_hypersphere(dim, n_samples, norm=1):
    if norm==1: # TODO ask question about that
        samples = np.random.rand(n_samples, dim)
        samples = samples / np.expand_dims(np.abs(samples).sum(1), 1)
        return samples
    elif norm==2:
        samples = np.random.randn(n_samples, dim)
        samples = samples / np.expand_dims(np.sqrt((samples ** 2).sum(1)), 1)
        return samples
    else:
        raise ValueError


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--agents", type=str, required=True, help="select agents appearing on the plot", nargs='+')
    parser.add_argument("-n", type=str, default='5', help="number of items in experiments plotted")
    parser.add_argument("-k", type=str, default='2', help="size of the assortments in experiments plotted")
    parser.add_argument("--horizon", type=str, default='100', help="horizon in experiments plotted")
    parser.add_argument("--regret_plot", type=int, default=1, help="whether or not to plot regret curve of experiments")
    parser.add_argument("--action_plot", type=int, default=0, help="whether or not to plot action selection analysis")
    args = parser.parse_args()
    experiment_base_name = '_'.join([args.n, args.k, args.horizon])
    experiments_to_plot = [agent_key + '_' + experiment_base_name for agent_key in args.agents]
    if args.regret_plot:
        print_regret(experiments_to_plot)
    if args.action_plot:
        print_actions(experiments_to_plot)
