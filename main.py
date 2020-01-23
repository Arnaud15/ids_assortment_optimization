from argparse import ArgumentParser
from env import AssortmentEnvironment
from agent import Agent, ThompsonSamplingAgent
from utils import print_run
from scipy.stats import uniform
import numpy as np

parser = ArgumentParser()
parser.add_argument("-n", type=int, default=5, help="number of items available")
parser.add_argument("-k", type=int, default=2, help="size of the assortments")
parser.add_argument("--agent", type=str, default="random", help="agent model to interact with the environment")
parser.add_argument("--nsim", type=int, default=1000, help="number of random simulations to carry out with agent")
parser.add_argument("--verbose", type=int, default=0, help="verbose level for simulations")


def run(envnmt, actor, n_steps, verbose=0):
    """
    :param envnmt:
    :param actor:
    :param n_steps:
    :param verbose:
    :return:
    """
    actor.reset()  # Resets the internal state of the agent at the start of simulations (prior beliefs, etc...)
    rewards = np.zeros(n_steps)
    obs = [0] * n_steps

    for ix in range(n_steps):
        assortment = actor.act()
        item_selected = envnmt.step(assortment)
        obs[ix] = (assortment, item_selected)
        reward = actor.update(item_selected)  # agent observes item selected, perceive reward and updates its beliefs
        rewards[ix] = reward

    if verbose == 5:
        # Print out more detailed information about the run, stored in the outputs folder
        print_run(envnmt, actor, n_steps, obs, rewards)

    print(
        f'simulation terminated successfully: {n_steps:d}steps, {rewards.sum():.2f} total reward, {rewards.mean():.2f} mean reward per step')
    return obs, rewards


if __name__ == "__main__":
    args = parser.parse_args()

    true_preferences = np.concatenate([uniform.rvs(size=args.n),
                                       np.array([1.])])
    print(true_preferences)
    env = AssortmentEnvironment(n=args.n, v=true_preferences)

    agent = ThompsonSamplingAgent(k=args.k, n=args.n)

    observations, rews = run(env, agent, n_steps=args.nsim, verbose=args.verbose)


else:
    import sys

    sys.exit("main should be used as a script")
