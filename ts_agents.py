from typing import Type
from collections import defaultdict
from env import act_optimally
from base_agents import BayesAgent, x_beta_sampling
import numpy as np


class EpochSamplingTS(BayesAgent):
    def __init__(self, k, n, sampling, **kwargs):
        BayesAgent.__init__(self, k, n)
        self.correlated_sampling = sampling

    def reset(self):
        self._n_is = np.ones(self.n_items)
        self._v_is = np.ones(self.n_items)
        self.epoch_ended = True
        self.current_action = None
        self.epoch_picks = defaultdict(int)
        self.data_stored = defaultdict(list)

    def act(self):
        if self.epoch_ended:
            self.data_stored["steps"].append(1)
            action = self.proposal()
        else:
            self.data_stored["steps"].append(0)
            assert self.current_action is not None
            action = self.current_action
        return action

    def sample_from_posterior(self, n_samples):
        return x_beta_sampling(
            a_s=self._n_is,
            b_s=self._v_is,
            correlated_sampling=self.correlated_sampling,
            n_samples=n_samples,
        )

    def proposal(self):
        posterior_belief = self.sample_from_posterior(1)
        action = act_optimally(
            np.squeeze(posterior_belief), top_k=self.subset_size
        )
        self.current_action = action
        return action

    def update_posterior(self, item_selected):
        try:
            item_selected = item_selected[0]
        except TypeError:
            assert isinstance(item_selected, int)
        if item_selected == self.n_items:  # picked up the outside option
            self.epoch_ended = True
            assert self.current_action is not None
            self._n_is[self.current_action] += 1
            for item_ix, n_picks in self.epoch_picks.items():
                self._v_is[item_ix] += n_picks
            self.epoch_picks = defaultdict(int)
        else:
            self.epoch_picks[item_selected] += 1
            self.epoch_ended = False
