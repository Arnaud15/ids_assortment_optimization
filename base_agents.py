import numpy as np
import abc
from scipy.stats import beta
from collections import defaultdict


class Agent(abc.ABC):
    def __init__(self, k, n):
        self.assortment_size = k
        self.n_items = n

    @abc.abstractmethod
    def act(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    def perceive_reward(self, item):
        """
        :param item:  index in [0, n-1], "no item" is index n
        :return: reward of 1. if any item is selected, 0. otherwise
        """
        return 1. if item < self.n_items else 0.

    @abc.abstractmethod
    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        return reward


class EpochSamplingAgent(abc.ABC):
    def __init__(self, k, n, horizon=None, correlated_sampling=False):
        self.assortment_size = k
        self.n_items = n
        self.epoch_ended = True
        self.current_action = self.n_items
        self.epoch_picks = defaultdict(int)
        self.posterior_parameters = [(1, 1) for _ in range(self.n_items)]
        self.correlated_sampling = correlated_sampling
        self.T = horizon

    def act(self):
        if self.epoch_ended:
            # print("Customer picked no item and we make a new proposal")
            action = self.proposal()
            # print(f"action picked is {action}")
        else:
            action = self.current_action
        return action

    def sample_from_posterior(self, n_samples):
        if not self.correlated_sampling:
            return np.array(
                [(1 / beta.rvs(a=a_, b=b_, size=n_samples)) - 1 for (a_, b_) in self.posterior_parameters]).T
        else:
            gaussian_means = np.expand_dims([b_ / a_ for (a_, b_) in self.posterior_parameters], 0)
            gaussian_stds = np.expand_dims(
                [np.sqrt(((b_ / a_) * ((b_ / a_) + 1)) / a_) for
                 (a_, b_) in self.posterior_parameters], 0)
            theta_sampled = np.random.randn(n_samples, 1)
            sample = gaussian_means + theta_sampled * gaussian_stds
            sample = np.clip(sample, a_min=0., a_max=1.0)
            # print(f"sampled posterior parameters are: {sample}")
            return sample

    def perceive_reward(self, item):
        """
        :param item:  index in [0, n-1], "no item" is index n
        :return: reward of 1. if any item is selected, 0. otherwise
        """
        return 1. if item < self.n_items else 0.

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        return reward

    @abc.abstractmethod
    def proposal(self):
        pass


class RandomAgent(Agent):
    def __init__(self, k, n):
        super().__init__(k, n)

    def act(self):
        return np.random.choice(np.arange(self.n_items, dtype=int), size=self.assortment_size, replace=False)

    def reset(self):
        pass

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        return reward
