from argparse import ArgumentParser
from env import AssortmentEnvironment
from base_agents import RandomAgent, OptimalAgent
from ts_agents import ThompsonSamplingAgent
from ids_agents import InformationDirectedSamplingAgent
from utils import print_actions, print_regret
from scipy.stats import uniform
import numpy as np
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("-n", type=int, default=5, help="number of items available")
parser.add_argument("-k", type=int, default=2, help="size of the assortments")
parser.add_argument("--horizon", type=int, default=50, help="number of random simulations to carry out with agent")
parser.add_argument("--nruns", type=int, default=2, help="number of random simulations to carry out with agent")
parser.add_argument("--verbose", type=int, default=0, help="verbose level for simulations")
parser.add_argument("--mode", type=str, default="regret", help="verbose level for simulations")

AGENTS = {"random": RandomAgent,
          "thompson_sampling": ThompsonSamplingAgent,
          "information_directed_sampling": InformationDirectedSamplingAgent}


def run(envnmt, actor, n_steps, verbose=0):
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
    if verbose:
        print(
            f'Simulation terminated successfully: {n_steps:d}steps, {rewards.sum():.2f} total reward, {rewards.mean():.2f} mean reward per step')
    return obs, rewards


if __name__ == "__main__":
    args = parser.parse_args()

    true_preferences = np.concatenate([uniform.rvs(size=args.n),
                                       np.array([1.])])
    # true_preferences = np.concatenate([np.array([0.1, 0.2, 0.5, 0.5, 0.3]),
    #                                    np.array([1.])])
    print(f"Initial preferences are: {true_preferences}.")
    print(f"Run type is {args.mode.upper()}")
    env = AssortmentEnvironment(n=args.n, v=true_preferences)

    experimental_results = {}
    for agent_name, agent_class in AGENTS.items():
        agent = agent_class(k=args.k, n=args.n)  # TODO init call might need to change
        agent_obs = []
        agent_rewards = np.zeros(args.horizon)
        for _ in range(args.nruns):
            if args.mode == 'regret':
                run_preferences = np.concatenate([uniform.rvs(size=args.n),
                                                  np.array([1.])])
                env.preferences = run_preferences
            obs_run, rewards_run = run(envnmt=env, actor=agent, n_steps=args.horizon, verbose=args.verbose)
            agent_obs += obs_run
            agent_rewards += rewards_run

        experimental_results[agent_name] = (agent_obs, agent_rewards / args.nruns)

    if args.mode == 'regret':
        print_regret(experimental_results,
                     true_preferences=true_preferences,
                     assortment_size=args.k,
                     n_steps=args.horizon)
    else:
        print_actions(experimental_results,
                      true_preferences=true_preferences)

else:
    import sys

    sys.exit("main should be used as a script")

# TODO: rewards =1 for each item + assumption that actions must be assortments of maximal size
