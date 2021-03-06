from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import beta
from collections import defaultdict
from args import PAPER_UNDEFINED_PRIOR


class BayesAgent(ABC):
    def __init__(self, k, n):
        self.subset_size = k
        self.n_items = n
        self.data_stored = defaultdict(list)

    @abstractmethod
    def act(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def sample_from_posterior(self, n_samples):
        """
        input is number of samples
        output of shape (n_samples, n_items)
         = sampled beliefs in (0, 1) for each sample, item
        """
        raise NotImplementedError()

    def stored_info(self):
        return self.data_stored

    @abstractmethod
    def update_posterior(self, items_selected):
        raise NotImplementedError()


class RandomAgent(BayesAgent):
    def __init__(self, k, n, **kwargs):
        super().__init__(k, n)

    def act(self):
        return np.random.choice(
            np.arange(self.n_items, dtype=int),
            size=self.subset_size,
            replace=False,
        )

    def reset(self):
        pass

    def update_posterior(self, item_selected):
        pass

    def sample_from_posterior(self, n_samples):
        return np.random.rand(n_samples, self.n_items)


class EpochSamplingAgent(BayesAgent, ABC):
    def __init__(
        self, k, n
    ):
        BayesAgent.__init__(self, k, n)
        self.epoch_ended = True
        self.current_action = None
        self.epoch_picks = defaultdict(int)

    def act(self):
        if self.epoch_ended:
            self.data_stored["steps"].append(1)
            action = self.proposal()
        else:
            self.data_stored["steps"].append(0)
            assert self.current_action is not None
            action = self.current_action
        return action

    @abstractmethod
    def proposal(self):
        pass


def x_beta_sampling(a_s, b_s, n_samples: int, correlated_sampling: bool) -> np.ndarray:
    """
    n_samples: how many samples to draw from the posterior
    returns: 2D array of shape (n_samples, N_items) of posterior samples
    """
    assert a_s.shape[0] == b_s.shape[0]
    assert len(a_s.shape) == 1
    assert len(b_s.shape) == 1
    if correlated_sampling:
        if PAPER_UNDEFINED_PRIOR:
            gaussian_means = b_s / a_s
            gaussian_stds = np.sqrt(b_s / a_s * ((b_s / a_s) + 1) / a_s)
        else:
            gaussian_means = b_s / (a_s - 1)
            gaussian_stds = np.sqrt(
                (b_s / (a_s - 1)) * ((b_s / (a_s - 1)) + 1) / (a_s - 2)
            )
        theta_sampled = np.random.randn(n_samples, 1)
        return gaussian_means.reshape(1, -1) + theta_sampled * gaussian_stds.reshape(1, -1)
    else:
        n_items = a_s.shape[0]
        return (
            1 / beta.rvs(a=a_s.reshape(1, -1), b=b_s.reshape(1, -1), size=(n_samples, n_items))
        ) - 1
