from argparse import ArgumentParser
from env import AssortmentEnvironment
from base_agents import RandomAgent
from ts_agents import ThompsonSamplingAgent, EpochSamplingTS
from ids_agents import InformationDirectedSamplingAgent, EpochSamplingIDS
from utils import save_experiment_data
from scipy.stats import uniform
import numpy as np
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--agent", type=str, required=True, help="choice of ts, ids, rd, ets, eids")
parser.add_argument("-n", type=int, default=5, help="number of items available")
parser.add_argument("-k", type=int, default=2, help="size of the assortments")
parser.add_argument("--horizon", type=int, default=500, help="number of random simulations to carry out with agent")
parser.add_argument("--nruns", type=int, default=2, help="number of random simulations to carry out with agent")
parser.add_argument("--fixed_preferences", type=int, default=0,
                    help="if you want episodes running with pre-defined preferences")
parser.add_argument("--ids_samples", type=int, default=250,
                    help="if you want episodes running with pre-defined preferences")

AGENTS = {"rd": RandomAgent,
          "ts": ThompsonSamplingAgent,
          "ids": InformationDirectedSamplingAgent,
          "ets": EpochSamplingTS,
          "eids": EpochSamplingIDS}

FIXED_PREFERENCES = np.concatenate([np.array([0.1, 0.2, 0.5, 0.5, 0.3]),
                                    np.array([1.])])


def run_episode(envnmt, actor, n_steps):
    """
    :param envnmt: instance from the AssortmentEnvironment class
    :param actor: instance from the Agent class
    :param n_steps: horizon of the run in the environment
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
        # if ix > (n_steps - 2):
        #     print("|" * 15)
        #     print(f"action is {np.arange(actor.n_items)[assortment]}")
        #     print("|" * 15)
        #     import ipdb;
        #     ipdb.set_trace()

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
    correlated_sampling = args.agent[-2:] == "cs"
    agent_key = args.agent[:-2] if correlated_sampling else args.agent

    agent_name = f"{args.agent}_{args.ids_samples}" if 'ids' in args.agent else args.agent

    exp_keys = [agent_name,
                args.n,
                args.k,
                args.horizon]
    exp_id = '_'.join([str(elt) for elt in exp_keys])
    print(f"Name of the agent is {exp_id}")

    # Agent init
    agent_class = AGENTS[agent_key]
    agent = agent_class(k=args.k,
                        n=args.n,
                        correlated_sampling=correlated_sampling,
                        horizon=args.horizon,
                        number_of_ids_samples=args.ids_samples)

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
        prefs_str = [f'{run_preference:.2f}' for run_preference in run_preferences]
        print(f'Initial preferences were :{prefs_str}')
        print(f'Best action was: {run_preferences.argsort()[-(args.k + 1):][::-1][1:]}')

        run_data["rewards"] = rewards_run
        run_data.update(summarize_run(obs_run))

        experiment_data.append(run_data)

    save_experiment_data(exp_id, experiment_data)
    print(f"Experiment successfully terminated")

else:
    import sys

    sys.exit("main should be used as a script")

# TODO: rewards =1 for each item + assumption that actions must be assortments of maximal size
