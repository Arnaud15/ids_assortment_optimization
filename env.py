import numpy as np


class AssortmentEnvironment(object):
    def __init__(self, n, v):
        self.items = np.arange(n + 1)
        self.n_items = n
        self.preferences = v  # preference are in (0, 1) for all items in [1, ..., N], 1 for "no item"

    def reset(self):
        if np.isinf(self.preferences).any():
            self.top_item = np.argmax(self.preferences)
        else:
            self.top_item = None

    def step(self, assortment):
        """
        :param assortment: array of K integers that specify the assortment, should be K distinct integers in [0, N-1]
        :return: obs: index of the item selected in [0, ..., n] -> n is when "no item" is selected
        """
        assert self.preferences[self.n_items] == 1.0
        if self.top_item is not None and self.top_item in assortment:
            return self.top_item
        else:
            possible_items = np.concatenate(
                [np.array([self.n_items], dtype=int), self.items[assortment]]
            )  # "no item" can always happen
            subset_preferences = self.preferences[possible_items]
            sum_preferences = subset_preferences.sum()
            probabilities = subset_preferences / sum_preferences
            return np.random.choice(possible_items, size=1, p=probabilities)[0]
