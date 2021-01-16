from ids_utils import (
    InformationDirectedSampler,
    ids_exact_action,
    greedy_ids_action,
    information_ratio,
    info_gain_step,
    delta_step,
)
from env import act_optimally, possible_actions
from base_agents import Agent, EpochSamplingAgent
import numpy as np
from run_utils import params_to_gaussian
from math import pi
from scipy.stats import beta, geom
import cvxpy as cp


class EpochSamplingCIDS(EpochSamplingAgent):
    def __init__(
        self, k, n, horizon, limited_prefs, n_samples, **kwargs,
    ):
        EpochSamplingAgent.__init__(
            self,
            k,
            n,
            horizon=horizon,
            sampling=0,
            limited_preferences=limited_prefs,
        )
        self.n_samples = n_samples

    def ts_cs_action(self):
        posterior_belief = self.sample_from_posterior(1)
        return act_optimally(
            np.squeeze(posterior_belief), top_k=self.assortment_size
        )

    def proposal(self):
        # self.prior_belief = self.sample_from_posterior(
        #     self.n_samples
        # )
        expected_rewards, stds = params_to_gaussian(self.posterior_parameters)

        entropies_start = 0.5 * np.log(2 * pi * np.exp(1) * stds ** 2)
        a_s = np.array([x[0] for x in self.posterior_parameters]).reshape(
            -1, 1
        )
        b_s = np.array([x[1] for x in self.posterior_parameters]).reshape(
            -1, 1
        )
        posterior_samples = (
            1 / beta.rvs(a=a_s, b=b_s, size=(a_s.shape[0], self.n_samples)) - 1
        )
        observations_samples = geom.rvs(1 / (posterior_samples + 1)) - 1
        new_posteriors = [
            [
                (
                    self.posterior_parameters[i][0] + 1,
                    self.posterior_parameters[i][1]
                    + observations_samples[i][j],
                )
                for j in range(self.n_samples)
            ]
            for i in range(self.n_items)
        ]
        new_entropies = [
            [
                0.5 * np.log(2 * pi * np.exp(1) * std ** 2)
                for std in params_to_gaussian(new_posteriors[i])[1]
            ]
            for i in range(self.n_items)
        ]
        new_entropies = np.array(new_entropies)
        new_entropies = new_entropies.mean(1)
        reductions = entropies_start - new_entropies

        optimistic_expectations = expected_rewards + 2 * stds
        ts_cs_action = act_optimally(np.squeeze(optimistic_expectations), top_k=self.assortment_size)
        # ts_cs_action = self.ts_cs_action()
        ts_cs_gain = reductions[ts_cs_action].sum()
        
        x = cp.Variable(self.n_items)
        objective = cp.Maximize(expected_rewards @ x)
        constraints = [
            0 <= x,
            x <= 1,
            cp.sum(x) == self.assortment_size,
            x @ reductions >= ts_cs_gain,
        ]
        prob = cp.Problem(objective, constraints,)
        prob.solve(solver="ECOS")

        try:
            action = np.random.choice(
                a=np.arange(self.n_items),
                p=np.abs(x.value) / np.abs(x.value).sum(),
                size=self.assortment_size,
                replace=False,
            )
        except ValueError:
            import ipdb
            ipdb.set_trace()
        # action = act_optimally(np.squeeze(x.value), top_k=self.assortment_size)
        self.current_action = action
        return action


class EpochSamplingIDS(EpochSamplingAgent):
    def __init__(
        self,
        k,
        n,
        horizon,
        limited_prefs,
        n_samples,
        info_type,
        objective,
        dynamics,
        scaling,
        regret_threshold,
        **kwargs,
    ):
        EpochSamplingAgent.__init__(
            self,
            k,
            n,
            horizon=horizon,
            sampling=0,
            limited_preferences=limited_prefs,
        )
        self.ids_sampler = InformationDirectedSampler(
            n_items=n,
            assortment_size=k,
            info_type=info_type,
            n_samples=n_samples,
            dynamics=dynamics,
        )
        self.fitted_scaler = 1.0
        self.scaling = scaling
        self.objective = objective
        self.regret_threshold = regret_threshold
        if self.objective == "exact":
            self.all_actions = np.array(
                possible_actions(self.n_items, self.assortment_size),
                dtype=int,
            )

    def proposal(self):
        self.prior_belief = self.sample_from_posterior(
            self.ids_sampler.n_samples
        )
        self.ids_sampler.update_belief(self.prior_belief)
        greedy_proposal = (
            np.sqrt(self.ids_sampler.delta_min) < self.regret_threshold
        )
        if greedy_proposal:
            assortment = act_optimally(
                np.squeeze(self.prior_belief.mean(0)),
                top_k=self.assortment_size,
            )
            g_approx = info_gain_step(
                action=assortment,
                sampled_preferences=self.prior_belief,
                actions_star=self.ids_sampler.actions_star,
                counts=self.ids_sampler.counts_star,
                thetas=self.ids_sampler.thetas_star,
            )
            g_approx = 1e-12 if g_approx < 1e-12 else g_approx
            d_approx = delta_step(
                action=assortment,
                sampled_preferences=self.prior_belief,
                r_star=self.ids_sampler.r_star,
            )
            rho_policy = 0.5
            ir_assortment = information_ratio(
                rho=rho_policy,
                d1=d_approx,
                d2=d_approx,
                g1=g_approx,
                g2=g_approx,
            )
            self.data_stored["greedy"].append(1)
        elif self.objective == "exact":
            assortment, ir_assortment, rho_policy = ids_exact_action(
                g_=self.ids_sampler.g_,
                d_=self.ids_sampler.d_,
                actions_set=self.all_actions,
                sampled_preferences=self.prior_belief,
                r_star=self.ids_sampler.r_star,
                actions_star=self.ids_sampler.actions_star,
                counts_star=self.ids_sampler.counts_star,
                thetas_star=self.ids_sampler.thetas_star,
            )
        else:
            assert self.objective == "lambda", "Choice of [exact, lambda]."
            if self.scaling == "autoreg":
                lambda_scaler = self.fitted_scaler
            elif self.scaling == "time":
                lambda_scaler = self.ids_sampler.lambda_algo * (
                    self.T - self.current_step
                )
            else:
                raise ValueError("Scaling: choice of [autoreg, time].")
            # print("check in the epoch setting")
            # print(self.ids_sampler.r_star)
            # print(
            #     self.ids_sampler.d_(
            #         np.arange(self.assortment_size),
            #         self.prior_belief,
            #         self.ids_sampler.r_star,
            #     )
            # )
            assortment, ir_assortment, rho_policy = greedy_ids_action(
                scaling_factor=lambda_scaler,
                g_=self.ids_sampler.g_,
                d_=self.ids_sampler.d_,
                sampled_preferences=self.prior_belief,
                r_star=self.ids_sampler.r_star,
                actions_star=self.ids_sampler.actions_star,
                counts_star=self.ids_sampler.counts_star,
                thetas_star=self.ids_sampler.thetas_star,
            )
            self.data_stored["greedy"].append(0)
        self.current_action = assortment
        self.fitted_scaler = ir_assortment
        self.data_stored["info_ratio"].append(ir_assortment)
        self.data_stored["entropy_a_star"].append(
            self.ids_sampler.a_star_entropy
        )
        self.data_stored["rho_policy"].append(rho_policy)
        self.data_stored["delta_min_2"].append(self.ids_sampler.delta_min ** 2)
        return assortment


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
