from collections import defaultdict
import numpy as np
from scipy.stats import beta
from base_agents import BayesAgent
from env import act_optimally
from icecream import ic


def beta_sampling(a_s, b_s, n_samples):
    assert a_s.shape[0] == b_s.shape[0]
    assert len(a_s.shape) == 1
    assert len(b_s.shape) == 1
    return beta.rvs(
        a=a_s.reshape(1, -1),
        b=a_s.reshape(1, -1),
        size=(n_samples, a_s.shape[0]),
    )


class ThompsonIDS(BayesAgent):
    def __init__(self, k, n, *args, **kwargs):
        super().__init__(k=k, n=n)

    def act(self):
        self.data_stored["steps"].append(1)
        act = np.random.choice(
            self.n_items, size=self.subset_size, replace=False
        )
        self.current_action = act
        return act

    def reset(self):
        self._n_is = np.ones(self.n_items)
        self._v_is = np.ones(self.n_items)
        self.epoch_ended = True
        self.current_action = None
        self.data_stored = defaultdict(list)
        self.current_action = None

    def sample_from_posterior(self, n_samples):
        return beta_sampling(
            a_s=self._n_is, b_s=self._v_is, n_samples=n_samples,
        )

    def update_posterior(self, items_selected):
        assert self.current_action is not None
        self._n_is[self.current_action] += 1
        self._v_is[items_selected] += 1
        # ic(items_selected)
        # ic(self._n_is, self._v_is)


class ThompsonSemit(BayesAgent):
    def __init__(self, k, n, *args, **kwargs):
        super().__init__(k=k, n=n)

    def act(self):
        self.data_stored["steps"].append(1)
        belief = self.sample_from_posterior(n_samples=1)
        act = act_optimally(belief=belief, top_k=self.subset_size)
        self.current_action = act
        return act

    def reset(self):
        self._n_is = np.ones(self.n_items)
        self._v_is = np.ones(self.n_items)
        self.epoch_ended = True
        self.current_action = None
        self.data_stored = defaultdict(list)
        self.current_action = None

    def sample_from_posterior(self, n_samples):
        return beta_sampling(
            a_s=self._n_is, b_s=self._v_is, n_samples=n_samples,
        )

    def update_posterior(self, items_selected):
        assert self.current_action is not None
        self._n_is[self.current_action] += 1
        self._v_is[items_selected] += 1
        # ic(self.current_action, items_selected)
        # ic(self._n_is, self._v_is)
