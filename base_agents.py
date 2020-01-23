import numpy as np
import abc
from utils import act_optimally


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


class OptimalAgent(Agent):
    def __init__(self, k, n, true_preferences):
        super().__init__(k, n)
        self.correct_belief = true_preferences

    def act(self):
        return act_optimally(self.correct_belief, top_k=self.assortment_size)

    def reset(self):
        pass

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        return reward
