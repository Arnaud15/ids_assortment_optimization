from argparse import ArgumentParser
from env import AssortmentEnvironment
from base_agents import RandomAgent, OptimalAgent
from ts_agents import ThompsonSamplingAgent
from ids_agents import InformationDirectedSamplingAgent
from utils import print_actions, print_regret, save_experiment_data
from scipy.stats import uniform
import numpy as np
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--agent", type=str, required=True, help="choice of ts, ids, rd")
parser.add_argument("-n", type=int, default=3, help="number of items available")
parser.add_argument("-k", type=int, default=2, help="size of the assortments")
parser.add_argument("--horizon", type=int, default=3, help="number of random simulations to carry out with agent")
parser.add_argument("--nruns", type=int, default=2, help="number of random simulations to carry out with agent")
parser.add_argument("--verbose", type=int, default=0, help="verbose level for simulations")
parser.add_argument("--fixed_preferences", type=int, default=0,
                    help="if you want episodes running with pre-defined preferences")

AGENT_IDS = {'ts': "thompson_sampling",
             'rd': "random",
             'ids': "information_directed_sampling"}

AGENTS = {"random": RandomAgent,
          "thompson_sampling": ThompsonSamplingAgent,
          "information_directed_sampling": InformationDirectedSamplingAgent}

FIXED_PREFERENCES = np.concatenate([np.array([0.1, 0.2, 0.5, 0.5, 0.3]),
                                    np.array([1.])])


def run_episode(envnmt, actor, n_steps):
    """
    :param envnmt: instance from the AssortmentEnvironment class
    :param actor: instance from the Agent class
    :param n_steps: horizon of the run in the environment
    :param verbose: control how much you want to print info
    :return: (observations history = list of (assortment one-hot array of size N+1, 0<=index<=N of item picked),
    rewards = historical rewards)
    """
    actor.reset()  # Resets the internal state of the agent at the start of simulations (prior beliefs, etc...)
    rewards = np.zeros(n_steps)
    obs = [0] * n_steps

    for ix in tqdm(range(n_steps)):
        assortment = actor.act()
        item_selected = envnmt.step(assortment)
        obs[ix] = (assortment, item_selected)
        reward = actor.update(item_selected)  # agent observes item selected, perceive reward and updates its beliefs
        rewards[ix] = reward

    return obs, rewards


def summarize_run(observations):
    run_assortments = sum([assortment for assortment, item_picked in observations])
    run_picks = {item_ix: 0 for item_ix in range(args.n + 1)}
    for assortment, item_picked in observations:
        run_picks[item_picked] += 1
    return {"assortments": run_assortments,
            "picks": run_picks}


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"arguments are {args}")
    print(f"Fixed preferences are: {FIXED_PREFERENCES}.")
    print(f"Preferences used are {'fixed' if args.fixed_preferences else 'random'}")

    # Experiment identifier in storage
    exp_keys = [args.agent,
                args.n,
                args.k,
                args.horizon]
    exp_id = '_'.join([str(elt) for elt in exp_keys])

    # Agent init
    agent_name = AGENT_IDS[args.agent]
    agent_class = AGENTS[agent_name]
    agent = agent_class(k=args.k, n=args.n)  # TODO init call might need to change

    experiment_data = []
    for _ in range(args.nruns):
        run_preferences = FIXED_PREFERENCES
        if not args.fixed_preferences:
            run_preferences = np.concatenate([uniform.rvs(size=args.n),
                                              np.array([1.])])
        env = AssortmentEnvironment(n=args.n, v=run_preferences)
        top_preferences = np.sort(run_preferences)[-(args.k + 1):]
        top_preferences = top_preferences / top_preferences.sum()
        expected_reward_from_best_action = top_preferences[:args.k].sum()

        run_data = {"best_reward": expected_reward_from_best_action}
        obs_run, rewards_run = run_episode(envnmt=env, actor=agent, n_steps=args.horizon)

        run_data["rewards"] = rewards_run
        run_data.update(summarize_run(obs_run))

        experiment_data.append(run_data)

    save_experiment_data(exp_id, experiment_data)
    # TODO test this
    # TODO launch this
    # TODO code printing results
    print(f"Experiment successfully terminated")

    # if args.mode == 'regret':
    #     print_regret(experimental_results,
    #                  true_preferences=true_preferences,
    #                  assortment_size=args.k,
    #                  n_steps=args.horizon)
    # else:
    #     print_actions(experimental_results,
    #                   true_preferences=true_preferences)

else:
    import sys

    sys.exit("main should be used as a script")

# TODO: rewards =1 for each item + assumption that actions must be assortments of maximal size
