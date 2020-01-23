from mcmc import sample_from_posterior
from scipy.stats import uniform
from utils import act_optimally
from base_agents import Agent
import numpy as np


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


# TODO finish this
class ApproximateThompsonSamplingAgent(Agent):
    def __init__(self, k, n):
        super().__init__(k, n)
        self.prior_belief = uniform.rvs(size=n)
        self.assortments_given = []
        self.item_picks = []
        self.inside_epoch_sampling = True

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
        if item_selected == self.n_items:
            self.inside_epoch_sampling = False

        reward = self.perceive_reward(item_selected)
        self.item_picks.append(item_selected)
        self.prior_belief = np.squeeze(sample_from_posterior(n_samples=1,
                                                             assortments=np.array(self.assortments_given),
                                                             item_picks=np.array(self.item_picks),
                                                             n_observations=len(self.item_picks),
                                                             n_items=self.n_items))
        return reward
