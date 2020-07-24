from env import act_optimally
from base_agents import Agent, EpochSamplingAgent
import numpy as np


class EpochSamplingTS(EpochSamplingAgent):
    def __init__(self, k, n, horizon, sampling, limited_prefs, **kwargs):
        EpochSamplingAgent.__init__(
            self,
            k,
            n,
            horizon=horizon,
            sampling=sampling,
            limited_preferences=limited_prefs,
        )

    def proposal(self):
        posterior_belief = self.sample_from_posterior(1)
        action = act_optimally(
            np.squeeze(posterior_belief), top_k=self.assortment_size
        )
        self.current_action = action
        return action


class SparseTS(Agent):
    def __init__(
        self, k, n, sampling, fallback_weight, p_optim=None, **kwargs,
    ):
        super().__init__(k, n)
        self.fallback_weight = fallback_weight
        self.sampling = sampling
        assert (
            self.sampling <= 1
        ), "Optimistic sampling currently not supported."
        self.optimism_probability = p_optim
        print(
            f"Agent with cs={self.sampling} and p_o={self.optimism_probability}"
        )
        self.reset()

    def reset(self):
        self.top_item_index = None
        # All items except:
        # the outside alternative (index N)
        # the fallback item (index 0)
        self.normal_items_indices = np.arange(1, self.n_items)
        self.top_item_posterior_probabilites = np.ones(self.n_items) / (
            self.n_items - 1
        )
        # We already know that
        # fallback item is not the top one
        self.top_item_posterior_probabilites[0] = 0.0
        assert self.top_item_posterior_probabilites.sum() > (1.0 - 1e-5)

    def sample_from_posterior(self, n_samples: int) -> np.ndarray:
        """
        :param n_samples: number of posterior samples desired
        :return samples: item preferences for each sample, of shape (n_samples, N)
        """
        samples = np.zeros(shape=(n_samples, self.n_items))
        if self.top_item_index is not None:
            # If we found the top item
            # Posterior is deterministic
            samples[:, self.top_item_index] = np.inf
            samples[:, 0] = self.fallback_weight
        elif not self.sampling:
            # Sample top item randomly
            top_item_guesses = np.random.choice(
                self.normal_items_indices,
                replace=True,
                size=n_samples,
                p=self.top_item_posterior_probabilites[1:],
            )
            samples[np.arange(n_samples), top_item_guesses] = np.inf
        else:
            # Sample for one item if it is optimal
            # Share this sample for all new items
            proba_item_is_optimal = (
                self.top_item_posterior_probabilites.max()
                if self.optimism_probability is None
                else self.optimism_probability
            )
            optimistic = np.random.rand() <= proba_item_is_optimal
            uncertain_items = self.normal_items_indices[
                self.top_item_posterior_probabilites[1:] > 0
            ]
            if optimistic:
                samples[:, uncertain_items] = np.inf
            else:
                samples[:, uncertain_items] = 0
        samples[:, 0] = self.fallback_weight
        return samples

    def act(self):
        posterior_belief = self.sample_from_posterior(n_samples=1)
        action = act_optimally(
            np.squeeze(posterior_belief), top_k=self.assortment_size
        )
        self.current_action = action
        assert 0 in action if (not self.sampling) else True
        assert (
            self.top_item_index in action
            if self.top_item_index is not None
            else True
        )
        return action

    def update(self, item_selected):
        if (item_selected > 0) and (item_selected < self.n_items):
            # Found the top item, which is the item selected
            self.top_item_index = item_selected
        else:
            # Top item was not in the assortment proposed
            self.top_item_posterior_probabilites[self.current_action] = 0
            self.top_item_posterior_probabilites = (
                self.top_item_posterior_probabilites
                / self.top_item_posterior_probabilites.sum()
            )
            non_zeros = self.top_item_posterior_probabilites[
                self.top_item_posterior_probabilites > 0
            ]
            assert non_zeros.std() < 1e-3
        reward = self.perceive_reward(item_selected)
        return reward
