import numpy as np
import logging
from collections import Counter


class AssortmentEnvironment(object):
    def __init__(self, n, v):
        self.items = np.arange(n + 1)
        self.n_items = n
        self.preferences = v
        self.preferences = (self.preferences * 100).astype(int) / 100
        self.counts = Counter()
        self.selects = Counter()
        logging.basicConfig(
            level=logging.DEBUG,
            filename="logs.log",
            format="%(levelname)s:%(message)s",
        )
        logging.info(f"prefs={self.preferences}")

    def reset(self):
        if np.isinf(self.preferences).any():
            self.top_item = np.argmax(self.preferences)
        else:
            self.top_item = None

    def step(self, assortment):
        """
        :param assortment: array of K integers that specify the assortment,
        should be K distinct integers in [0, N-1]
        :return: obs: index of the item selected in [0, ..., n]
        -> n is when "no item" is selected
        """
        assert self.preferences[self.n_items] == 1.0
        if self.top_item is not None and self.top_item in assortment:
            return self.top_item
        else:
            possible_items = np.concatenate(
                [np.array([self.n_items], dtype=int), self.items[assortment]]
            )  # "no item" can always happen
            for item in possible_items:
                self.counts[item] += 1
            subset_preferences = self.preferences[possible_items]
            sum_preferences = subset_preferences.sum()
            probabilities = subset_preferences / sum_preferences
            selected = np.random.choice(possible_items, size=1, p=probabilities)[0]
            self.selects[selected] += 1
            return selected


def act_optimally(belief, top_k):
    noise_breaking_ties = np.random.randn(*belief.shape) * 1e-5
    belief += noise_breaking_ties
    if len(belief.shape) <= 1:
        return np.sort(np.argpartition(belief, -top_k)[-top_k:])
    else:
        return np.sort(
            np.argpartition(belief, -top_k, axis=1)[:, -top_k:], axis=1
        )


def possible_actions(n_items, assortment_size):
    assert assortment_size >= 1
    if assortment_size == 1:
        return [[i] for i in range(n_items)]
    else:
        prev_lists = possible_actions(n_items, assortment_size - 1)
        return [
            prev_list + [i]
            for prev_list in prev_lists
            for i in range(prev_list[-1] + 1, n_items)
        ]
