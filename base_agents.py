import numpy as np
import abc
from scipy.stats import beta
from collections import defaultdict
from args import (
    BAD_ITEM_CONSTANT,
    PAPER_EXPLORATION_BONUS,
    PAPER_UNDEFINED_PRIOR,
    TOP_ITEM_CONSTANT,
    BETA_RVS,
)


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

    @abc.abstractmethod
    def sample_from_posterior(self, n_samples):
        """
        input is number of samples
        output of shape (n_samples, n_items)
         = sampled beliefs in (0, 1) for each sample, item
        """
        pass

    def perceive_reward(self, item):
        """
        :param item:  index in [0, n-1], "no item" is index n
        :return: reward of 1. if any item is selected, 0. otherwise
        """
        return 1.0 if item < self.n_items else 0.0

    @abc.abstractmethod
    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        return reward


class RandomAgent(Agent):
    def __init__(self, k, n, **kwargs):
        super().__init__(k, n)

    def act(self):
        return np.random.choice(
            np.arange(self.n_items, dtype=int),
            size=self.assortment_size,
            replace=False,
        )

    def reset(self):
        pass

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        return reward

    def sample_from_posterior(self, n_samples):
        return np.random.rand(n_samples, self.n_items)


class EpochSamplingAgent(Agent, abc.ABC):
    def __init__(
        self,
        k,
        n,
        horizon=None,
        correlated_sampling=False,
        limited_preferences=0,
    ):
        Agent.__init__(self, k, n)
        self.correlated_sampling = correlated_sampling
        self.T = horizon
        self.first_item_best = True if limited_preferences else False
        print(f"Agent believes first=best? {self.first_item_best}")
        self.reset()

    def act(self):
        if self.epoch_ended:
            action = self.proposal()
        else:
            action = self.current_action
        return action

    def sample_from_posterior(self, n_samples):
        """
        n_samples: how many samples to draw from the posterior
        returns: 2D array of shape (n_samples, N_items) of posterior samples
        """
        if PAPER_EXPLORATION_BONUS:
            gaussian_means = np.expand_dims(
                [b_ / a_ for (a_, b_) in self.posterior_parameters], 0
            )
            # Using their extra exploration bonus
            gaussian_stds = np.expand_dims(
                [
                    np.sqrt(50 * (b_ / a_) * ((b_ / a_) + 1) / a_)
                    + 75 * np.sqrt(np.log(self.T * self.assortment_size)) / a_
                    for (a_, b_) in self.posterior_parameters
                ],
                0,
            )
        else:
            # Simply approximating the 1 / Beta(alpha, beta) - 1
            # by a normal distribution
            if PAPER_UNDEFINED_PRIOR:
                # In the paper, they make a strong approximation as 1/beta - 1
                # has no well-defined mean / variance
                gaussian_stds = np.expand_dims(
                    [
                        np.sqrt(b_ / a_ * ((b_ / a_) + 1) / a_)
                        for (a_, b_) in self.posterior_parameters
                    ],
                    0,
                )
                gaussian_means = np.expand_dims(
                    [b_ / a_ for (a_, b_) in self.posterior_parameters], 0
                )
            else:
                # In our setting, we start with a (3, 3) prior for each item
                #  having a well-defined mean / variance
                gaussian_stds = np.expand_dims(
                    [
                        np.sqrt(
                            (b_ / (a_ - 1)) * ((b_ / (a_ - 1)) + 1) / (a_ - 2)
                        )
                        for (a_, b_) in self.posterior_parameters
                    ],
                    0,
                )
                gaussian_means = np.expand_dims(
                    [b_ / (a_ - 1) for (a_, b_) in self.posterior_parameters],
                    0,
                )
        theta_sampled = (
            np.random.randn(n_samples, 1)
            if self.correlated_sampling
            else np.random.randn(n_samples, self.n_items)
        )
        # 1 / Beta(alpha, beta) - 1 prior from the Columbia paper
        # Given up on this: slow and not so good
        sample = gaussian_means + theta_sampled * gaussian_stds
        if BETA_RVS and (not self.correlated_sampling):
            return np.array(
                [
                    (1 / beta.rvs(a=a_, b=b_, size=n_samples)) - 1
                    for (a_, b_) in self.posterior_parameters
                ]
            ).T
        else:
            return sample

    def reset(self):
        self.epoch_ended = True
        self.current_action = self.n_items
        self.epoch_picks = defaultdict(int)
        if PAPER_EXPLORATION_BONUS or PAPER_UNDEFINED_PRIOR:
            self.posterior_parameters = [(1, 1) for _ in range(self.n_items)]
        else:
            self.posterior_parameters = [(3, 3) for _ in range(self.n_items)]
        if self.first_item_best:
            # Corrected prior when we know that a single item is "worth it"
            self.posterior_parameters = [
                (a_, b_ * BAD_ITEM_CONSTANT)
                for (a_, b_) in self.posterior_parameters
            ]
            self.posterior_parameters[0] = (1e5, 1e5 * TOP_ITEM_CONSTANT)

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        if item_selected == self.n_items:
            self.epoch_ended = True
            n_is = [
                int(ix in self.current_action) for ix in range(self.n_items)
            ]
            v_is = [self.epoch_picks[i] for i in range(self.n_items)]
            self.posterior_parameters = [
                (a + n_is[ix], b + v_is[ix])
                for ix, (a, b) in enumerate(self.posterior_parameters)
            ]
            self.epoch_picks = defaultdict(int)
        else:
            self.epoch_picks[item_selected] += 1
            self.epoch_ended = False
        return reward

    @abc.abstractmethod
    def proposal(self):
        pass
