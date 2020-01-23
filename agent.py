import numpy as np
from mcmc import sample_from_posterior
from scipy.stats import uniform
from utils import act_optimally


class Agent(object):
    def __init__(self, k, n):
        self.assortment_size = k
        self.n_items = n

    def act(self):
        return np.random.choice(np.arange(self.n_items, dtype=int), size=self.assortment_size, replace=False)

    def reset(self):
        pass

    def perceive_reward(self, item):
        """
        :param item:  index in [0, n-1], "no item" is index n
        :return: reward of 1. if any item is selected, 0. otherwise
        """
        return 1. if item < self.n_items else 0.

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        return reward


class ThompsonSamplingAgent(Agent):
    def __init__(self, k, n):
        super().__init__(k, n)
        self.prior_belief = uniform.rvs(size=n)
        self.assortments_given = []
        self.item_picks = []

    def act(self):
        action = act_optimally(self.prior_belief, top_k=self.assortment_size)
        assortment = np.zeros(self.n_items + 1)
        assortment[self.n_items] = 1.
        for item in action:
            assortment[item] = 1.
        self.assortments_given.append(assortment)
        return action

    def reset(self):
        self.prior_belief = uniform.rvs(size=self.n_items)
        self.assortments_given = []
        self.item_picks = []

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        self.item_picks.append(item_selected)
        self.prior_belief = np.squeeze(sample_from_posterior(n_samples=1,
                                                             assortments=np.array(self.assortments_given),
                                                             item_picks=np.array(self.item_picks),
                                                             n_observations=len(self.item_picks),
                                                             n_items=self.n_items))
        return reward
