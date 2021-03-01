from ids_utils import (
    approximate_info_gain,
    compute_info_gains,
    simplex_action_selection,
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
import logging


def logar(arr):
    return (arr * 1000).astype(int) / 1000

class EpochSamplingCIDS(EpochSamplingAgent):
    def __init__(
        self,
        k,
        n,
        horizon,
        limited_prefs,
        n_samples,
        info_type,
        frequentist,
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
        self.n_samples = n_samples
        assert info_type in {"variance", "gain"}
        self.info_type = info_type
        assert isinstance(frequentist, bool)
        self.frequentist = frequentist
        logging.basicConfig(
            level=logging.DEBUG,
            filename="logs.log",
            format="%(levelname)s:%(message)s",
        )
        logging.info(
            f"delta+ {self.frequentist}, approx_info_gain {self.info_type=='gain'}"
        )
        self.c = 1

    def ts_cs_action(self):
        posterior_belief = self.sample_from_posterior(1)
        return act_optimally(
            np.squeeze(posterior_belief), top_k=self.assortment_size
        )

    def proposal(self):
        # expected_rewards, stds = params_to_gaussian(self.posterior_parameters)
        # expected_rewards = np.minimum(expected_rewards, 1.0)

        posterior_belief = self.sample_from_posterior(self.n_samples)
        sorted_beliefs = np.sort(posterior_belief, axis=1)
        thresholds = sorted_beliefs[:, -self.assortment_size].reshape(-1, 1)

        best_actions = sorted_beliefs[:, -self.assortment_size :]
        sum_rewards_best = best_actions.sum(1)
        r_star = sum_rewards_best.mean()

        expected_rewards = posterior_belief.mean(0)
        # min_rew = expected_rewards.min() / 1e5
        # expected_rewards += np.random.rand(expected_rewards.shape[0]) * min_rew
        mask = posterior_belief >= thresholds
        p_star = mask.sum(0) / mask.shape[0]
        if_star = (posterior_belief * mask).sum(0) / (mask.sum(0) + 1e-12)
        # else_star = (posterior_belief * (1 - mask)).sum(0) / (
        #     (1 - mask).sum(0) + 1e-12
        # )
        # variances = (
        #     p_star * (if_star - expected_rewards) ** 2
        #     + (1 - p_star) * (else_star - expected_rewards) ** 2
        # )
        variances = p_star * (if_star - expected_rewards) ** 2
            # posterior_belief = self.sample_from_posterior(self.n_samples)
            # sorted_beliefs = np.sort(posterior_belief, axis=1)
            # thresholds = sorted_beliefs[:, -self.assortment_size].reshape(-1, 1)
            # mask = posterior_belief >= thresholds
            # p_star = mask.sum(0) / mask.shape[0]
            # variances *= p_star
        variances = np.maximum(variances, 1e-12)
        # a_star_t = np.sort(expected_rewards)[-self.assortment_size]
        # a_s = self.posterior_parameters[0]
        # b_s = self.posterior_parameters[1]
        # ps = beta.cdf(1 / (a_star_t + 1), a=a_s, b=b_s)
        # entropies_start = -(
        #     ps * np.log(np.maximum(ps, 1e-12))
        #     + (1 - ps) * np.log(np.maximum(1 - ps, +1e-12))
        # )
        # posterior_samples = 1 / beta.rvs(a=a_s, b=b_s) - 1
        # new_as = np.ones(self.n_items)
        # new_as += a_s
        # new_bs = (geom.rvs(1 / (posterior_samples + 1)) - 1) + b_s
        # new_ps = beta.cdf(1 / (a_star_t + 1), a=new_as, b=new_bs)
        # new_entropies = -(
        #     new_ps * np.log(np.maximum(new_ps, 1e-12))
        #     + (1 - new_ps) * np.log(np.maximum(1 - new_ps, +1e-12))
        # )
        # reductions = np.maximum(entropies_start - new_entropies, 1e-8)

        x = cp.Variable(self.n_items, pos=True)
        # deltas = cp.Parameter(self.n_items, pos=True)
        rewards = cp.Parameter(self.n_items,)
        gains = cp.Parameter(self.n_items, pos=True)
        # exp_regret = r_star - x @ rewards
        deltas = r_star - x @ rewards
        exp_gain = x @ gains
        information_ratio = cp.quad_over_lin(deltas, exp_gain)
        objective = cp.Minimize(information_ratio)
        constraints = [0 <= x, x <= 1, cp.sum(x) == self.assortment_size]
        prob = cp.Problem(objective, constraints,)
        rewards.value = expected_rewards
        gains.value = variances

        try:
            prob.solve(solver="ECOS")
            zeros_index = (x.value < 1e-3)
            ones_index = (x.value > 1 - 1e-3)
            nzeros = zeros_index.sum()
            nones = ones_index.sum()
            nitems = x.value.shape[0]
            logging.debug(
                f"{nitems - nones - nzeros} nstrict, {nones} ones, {nzeros} zeroes, {nitems} total items"
            )
            if (nitems - nones - nzeros) == 2:
                all_items = np.arange(nitems)
                strict_items = all_items[~np.bitwise_or(zeros_index, ones_index)]
                probas = x.value[~np.bitwise_or(zeros_index, ones_index)]
                assert strict_items.shape[0] == 2, strict_items
                assert probas.shape[0] == 2, probas
                # 2 items to randomize the selection over
                logging.debug(
                        f"items: {strict_items}, with probas: {probas}",
                )
                rho = probas[0]
                u = np.random.rand()
                if rho <= u:
                    remaning_item = strict_items[0]
                else:
                    remaning_item = strict_items[1]
                action = np.sort(np.concatenate([act_optimally(x.value,top_k=self.assortment_size - 1), np.array([remaning_item])]))
            else:
                action = act_optimally(x.value, top_k=self.assortment_size)
            if self.c % 5 == 121234:
                logging.debug(
                    f"a:{action},x:{(100 * x.value).astype(int)},rew:{(100 * expected_rewards).astype(int)},gain:{(100 * np.sqrt(variances)).astype(int)}"
                )
                logging.debug(
                        f"if_optimal: {if_star}, rew:{logar(expected_rewards)}, probas: {logar(p_star)}",
                )
                logging.debug(
                        f"if_optimal: {logar(if_star)}, rew:{logar(expected_rewards)}, probas: {logar(p_star)}",
                )
                logging.debug(
                    f"n{self.posterior_parameters[0]}, v{self.posterior_parameters[1] / self.posterior_parameters[0]},"
                )
                logging.debug(
                    f"obj{prob.value}"
                )
        except cp.SolverError:
            logging.warning("solver error")
            posterior_belief = self.sample_from_posterior(1)
            action = act_optimally(
                np.squeeze(posterior_belief), top_k=self.assortment_size
            )
        except TypeError:
            logging.warning("solver error")
            posterior_belief = self.sample_from_posterior(1)
            action = act_optimally(
                np.squeeze(posterior_belief), top_k=self.assortment_size
            )

        self.current_action = action
        self.c += 1
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
