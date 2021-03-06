import numpy as np
import logging
from collections import namedtuple

from abc import ABC, abstractmethod
import warnings


StepInfo = namedtuple("Info", ["obs", "reward"])


class CombEnv(ABC):
    def __init__(self, model_params):
        self._theta = model_params
        self.n_items = model_params.shape[0]
        self.items = np.arange(self.n_items)
        assert(np.all(self._theta > 0.0))
        self._counts = np.zeros(self.n_items)
        self._selections = np.zeros(self.n_items)

    def reset(self):
        if np.isinf(self._theta).any():
            self.top_item = np.argmax(self._theta)
        else:
            self.top_item = None

    @property
    def counts(self):
        return self._counts

    @counts.setter
    def counts(self, items_proposed):
        self._counts[items_proposed] += 1

    @property
    def selections(self):
        return self._selections

    @selections.setter
    def selections(self, items_selected):
        self._selections[items_selected] += 1

    @abstractmethod
    def step(self, action) -> StepInfo:
        raise NotImplementedError()

    @abstractmethod
    def expected_reward(self, action):
        raise NotImplementedError()

    def r_star_from_subset_size(self, k: int) -> float:
        """
        Computes and returns the Expected reward from best subset
        """
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            top_subset = act_optimally(self._theta, top_k=k)
        return self.expected_reward(top_subset)


class BernoulliSemi(CombEnv):
    def __init__(self, item_probas):
        super().__init__(model_params=item_probas)
        logging.info(
            f"Instantiated Bernoulli Semi Bandit Env with item preferences ={self._theta}"
        )

    def step(self, subset):
        self.counts = subset
        items_ok = np.random.rand(self.n_items) <= self._theta
        subset_ok = items_ok[subset]
        items_selected = subset[subset_ok]
        self.selections = items_selected
        return StepInfo(
            obs=items_selected, reward=self.expected_reward(subset)
        )

    def expected_reward(self, subset):
        subset_probas = self._theta[subset]
        return subset_probas.mean()


class AssortmentEnvironment(CombEnv):
    def __init__(self, item_prefs):
        super().__init__(model_params=item_prefs)
        logging.info(
            f"Instantiated Assortment Env with item preferences ={self._theta}"
        )

    @CombEnv.selections.setter
    def selections(self, item_selected):
        # overridding
        if item_selected < self.n_items:  # outside option not picked
            self._selections[item_selected] += 1

    def step(self, assortment):
        """
        :param assortment: array of K integers that specify the assortment,
        should be K distinct integers in [0, N-1]
        :return: obs: index of the item selected in [0, ..., n]
        -> n is when "no item" is selected
        """
        self.counts = assortment
        subset_preferences = self._theta[assortment]
        sum_preferences = subset_preferences.sum()
        proba_any_picked = sum_preferences / (1.0 + sum_preferences)
        outside_option_selected = np.random.rand() > proba_any_picked
        if not outside_option_selected:
            selected = np.random.choice(
                assortment, p=subset_preferences / sum_preferences, size=1
            )
        else:
            selected = (
                self.n_items
            )  # the outside option has index N, available items have indexes 0, ..., N-1
        self.selections = selected
        return StepInfo(obs=selected, reward=proba_any_picked)

    def expected_reward(self, assortment):
        subset_preferences = self._theta[assortment]
        sum_preferences = subset_preferences.sum()
        return sum_preferences / (1.0 + sum_preferences)


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
