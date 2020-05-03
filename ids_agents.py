from ids_utils import (
    InformationDirectedSampler,
    ids_action_selection_numba,
    ids_action_selection_approximate,
    greedy_ids_action_selection,
)
from env import act_optimally, possible_actions
from base_agents import Agent, EpochSamplingAgent
import numpy as np


class EpochSamplingIDS(EpochSamplingAgent):
    def __init__(
        self,
        k,
        n,
        correlated_sampling,
        limited_prefs,
        n_samples,
        info_type,
        action_type,
        scaling_factor=0.0,
        **kwargs,
    ):
        EpochSamplingAgent.__init__(
            self,
            k,
            n,
            horizon=None,
            correlated_sampling=False,
            limited_preferences=limited_prefs,
        )
        self.ids_sampler = InformationDirectedSampler(
            assortment_size=k, info_type=info_type, n_samples=n_samples
        )
        self.action_selection = action_type
        self.scaling_factor = scaling_factor
        print(
            f"Action selection+{self.action_selection}"
        )
        print(f"scaling factor: {self.scaling_factor}"
        if self.action_selection != "greedy":
            self.all_actions = np.array(
                possible_actions(self.n_items, self.assortment_size), dtype=int
            )

    def proposal(self):
        self.prior_belief = self.sample_from_posterior(
            self.ids_sampler.n_samples
        )
        self.ids_sampler.update_belief(self.prior_belief)
        if self.action_selection == "exact":
            action = np.array(
                ids_action_selection_numba(
                    g_=self.ids_sampler.g_,
                    actions_set=self.all_actions,
                    sampled_preferences=self.prior_belief,
                    r_star=self.ids_sampler.r_star,
                    actions_star=self.ids_sampler.actions_star,
                    counts_star=self.ids_sampler.counts_star,
                    thetas_star=self.ids_sampler.thetas_star,
                )
            )
        elif self.action_selection == "approximate":
            action = np.array(
                ids_action_selection_approximate(
                    g_=self.ids_sampler.g_,
                    actions_set=self.all_actions,
                    sampled_preferences=self.prior_belief,
                    r_star=self.ids_sampler.r_star,
                    actions_star=self.ids_sampler.actions_star,
                    counts_star=self.ids_sampler.counts_star,
                    thetas_star=self.ids_sampler.thetas_star,
                )
            )
        elif self.action_selection == "greedy":
            action = np.array(
                greedy_ids_action_selection(
                    g_=self.ids_sampler.g_,
                    scaling_factor=self.scaling_factor,
                    sampled_preferences=self.prior_belief,
                    r_star=self.ids_sampler.r_star,
                    actions_star=self.ids_sampler.actions_star,
                    counts_star=self.ids_sampler.counts_star,
                    thetas_star=self.ids_sampler.thetas_star,
                )
            )
        else:
            raise ValueError(
                "Must be one of (exact | approximate | greedy)"
            )
        self.current_action = action
        return action


def best_mixture(delta, gain):
    probs = np.linspace(0, 1, num=50)

    delta_p = np.expand_dims(np.outer(probs, delta), axis=2)
    gain_p = np.expand_dims(np.outer(probs, gain), axis=2)

    delta_1p = np.expand_dims(np.outer(1 - probs, delta), axis=1)
    gain_1p = np.expand_dims(np.outer(1 - probs, gain), axis=1)

    numerator = (delta_p + delta_1p) ** 2
    denominator = gain_p + gain_1p
    denominator[denominator < 1e-12] = 1e-12
    ratio = np.divide(numerator, denominator)

    min_ratio = ratio.min()
    index = np.argwhere(ratio == min_ratio)
    prob_ix, a1_ix, a2_ix = index[np.random.randint(index.shape[0]), :]
    return probs[prob_ix], a1_ix, a2_ix


def sparse_information_ratio(
    min_new_proposed, max_new_proposed, n_candidates, info_type, p0
):
    n_news = np.arange(min_new_proposed, max_new_proposed + 1)
    alphas = n_news / n_candidates
    deltas = 1 - np.concatenate(
        [alphas[1:], alphas[:-1] + p0 * (1 - alphas[:-1])], axis=0
    )
    if info_type == "variance":
        gains = alphas * (1 - alphas) ** 2 + (1 - alphas) * alphas ** 2
        gains = np.concatenate([gains[1:], gains[:-1] * (1 - p0) ** 2], axis=0)

    if info_type == "gain":
        alphas[alphas < 1e-12] = 1e-12
        alphas[alphas > 1.0 - 1e-12] = 1.0 - 1e-12
        gains = alphas * np.log(n_candidates) - (1 - alphas) * np.log(
            1 - alphas
        )
        assert not np.isinf(gains).any()
        gains = np.concatenate([gains[1:], gains[:-1]], axis=0)
    assert deltas.shape[0] == 2 * (max_new_proposed - min_new_proposed)
    return deltas, gains


class SparseIDS(Agent):
    def __init__(
        self, k, n, fallback_proba, fallback_weight, info_type, **kwargs
    ):
        super().__init__(k, n)
        self.info_type = info_type
        assert info_type in {"variance", "gain"}
        self.fallback_proba = fallback_proba
        self.fallback_weight = fallback_weight
        assert fallback_proba < 1.0 and fallback_proba > 0.0
        assert (
            np.abs(fallback_proba - fallback_weight / (1.0 + fallback_weight))
            < 1e-5
        )
        self.reset()

    def reset(self):
        self.top_item_index = None
        self.normal_items_indices = np.arange(1, self.n_items, dtype=int)
        self.possibly_top = np.ones(self.n_items, dtype=int)
        self.possibly_top[0] = 0
        self.to_assess = self.possibly_top.sum()

    def sample_from_posterior(self, n_samples: int) -> np.ndarray:
        """
        :param n_samples: number of posterior samples desired
        :return samples: item preferences for each sample, size (n_samples, N)
        """
        samples = np.zeros(shape=(n_samples, self.n_items))
        if self.top_item_index is not None:
            # If we found the top item
            # Posterior is deterministic
            samples[:, self.top_item_index] = np.inf
            samples[:, 0] = self.fallback_weight
        else:
            # Sample top item randomly
            top_item_guesses = np.random.choice(
                self.normal_items_indices,
                replace=True,
                size=n_samples,
                p=self.possibly_top[1:] / self.possibly_top[1:].sum(),
            )
            samples[np.arange(n_samples), top_item_guesses] = np.inf
        samples[:, 0] = self.fallback_weight
        return samples

    def update(self, item_selected):
        if (item_selected > 0) and (item_selected < self.n_items):
            # Found the top item, which is the item selected
            self.top_item_index = item_selected
        else:
            # Top item was not in the assortment proposed
            self.possibly_top[self.current_action] = 0
            self.to_assess = self.possibly_top.sum()
        reward = self.perceive_reward(item_selected)
        return reward

    def act(self):
        action = self.action_selection()
        self.current_action = action
        assert (
            self.top_item_index in action
            if self.top_item_index is not None
            else True
        )
        return action

    def action_selection(self):
        if self.top_item_index is None:
            fallback_taken, n_new = self.optimal_ids_action_parameters()
            action = self.sample_from_params(fallback_taken, n_new)
            return action
        else:
            return act_optimally(
                np.squeeze(self.sample_from_posterior(1)),
                top_k=self.assortment_size,
            )

    def optimal_ids_action_parameters(self):
        min_new = max(
            0, self.assortment_size - (self.n_items - self.to_assess)
        )
        max_new = min(self.assortment_size, self.to_assess)

        deltas, gains = sparse_information_ratio(
            min_new_proposed=min_new,
            max_new_proposed=max_new,
            n_candidates=self.to_assess,
            info_type=self.info_type,
            p0=self.fallback_proba,
        )  # double the size

        prob, a1_raw, a2_raw = best_mixture(delta=deltas, gain=gains)

        a1_f = 1 if a1_raw >= (max_new - min_new) else 0
        a1 = (a1_f, min_new + (a1_raw + 1) // 2 + 1 - 2 * a1_f)
        a2_f = 1 if a2_raw >= (max_new - min_new) else 0
        a2 = (a2_f, min_new + (a2_raw + 1) // 2 + 1 - 2 * a2_f)
        assert (a1[1] >= min_new) and (a1[1] <= max_new)
        assert (a2[1] >= min_new) and (a2[1] <= max_new)
        return a1 if np.random.rand() < prob else a2

    def sample_from_params(self, fallback_taken, n_new):
        assert (n_new + fallback_taken) <= self.assortment_size
        new_items = np.random.choice(
            self.normal_items_indices,
            replace=False,
            size=n_new,
            p=self.possibly_top[1:] / self.possibly_top[1:].sum(),
        )
        old_items = (
            np.random.choice(
                self.normal_items_indices,
                replace=False,
                size=self.assortment_size - n_new - fallback_taken,
                p=(1 - self.possibly_top[1:])
                / (1 - self.possibly_top[1:]).sum(),
            )
            if self.assortment_size - n_new - fallback_taken
            else np.array([], dtype=int)
        )
        action = np.concatenate(
            [
                old_items,
                new_items,
                np.array([0], dtype=int)
                if fallback_taken
                else np.array([], dtype=int),
            ],
            axis=0,
        )
        return action
