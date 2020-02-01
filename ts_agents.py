from mcmc import sample_from_posterior
from scipy.stats import uniform
from utils import act_optimally
from base_agents import Agent, EpochSamplingAgent
import numpy as np
from collections import defaultdict


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


class ApproximateThompsonSamplingAgent(EpochSamplingAgent):
    def __init__(self, k, n):
        super().__init__(k, n)

    def proposal(self):
        posterior_belief = self.sample_from_posterior(1)
        action = act_optimally(np.squeeze(posterior_belief), top_k=self.assortment_size)
        self.current_action = action
        return action

    def reset(self):
        self.epoch_ended = True
        self.current_action = self.n_items
        self.epoch_picks = defaultdict(int)
        self.posterior_parameters = [(1, 1) for _ in range(self.n_items)]

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        if item_selected == self.n_items:
            self.epoch_ended = True
            n_is = [int(ix in self.current_action) for ix in range(self.n_items)]
            v_is = [self.epoch_picks[i] for i in range(self.n_items)]
            self.posterior_parameters = [(a + n_is[ix], b + v_is[ix]) for ix, (a, b) in
                                         enumerate(self.posterior_parameters)]
            self.epoch_picks = defaultdict(int)
        else:
            self.epoch_picks[item_selected] += 1
            self.epoch_ended = False
        return reward
