from ids_utils import (
    compute_info_gains,
    InformationDirectedSampler,
    ids_exact_action,
    information_ratio,
    info_gain_step,
    delta_step,
)
from env import act_optimally, possible_actions
from base_agents import EpochSamplingAgent, x_beta_sampling
import numpy as np
from run_utils import params_to_gaussian
from collections import defaultdict
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
        n_samples,
        info_type,
        **kwargs,
    ):
        EpochSamplingAgent.__init__(
            self,
            k,
            n,
        )
        self.n_samples = n_samples
        assert info_type in {"variance", "gain"}
        self.info_type = info_type
        self._n_is = np.ones(self.n_items)
        self._v_is = np.ones(self.n_items)

    def reset(self):
        self._n_is = np.ones(self.n_items)
        self._v_is = np.ones(self.n_items)
        self.epoch_ended = True
        self.current_action = None
        self.epoch_picks = defaultdict(int)
        self.c = 0

    def sample_from_posterior(self, n_samples):
            return x_beta_sampling(a_s=self._n_is, b_s=self._v_is, correlated_sampling=False, n_samples=n_samples)

    def proposal(self):
        posterior_belief = self.sample_from_posterior(self.n_samples)
        sorted_beliefs = np.sort(posterior_belief, axis=1)
        thresholds = sorted_beliefs[:, -self.subset_size].reshape(-1, 1)

        best_actions = sorted_beliefs[:, -self.subset_size :]
        sum_rewards_best = best_actions.sum(1)
        r_star = sum_rewards_best.mean()

        expected_rewards = posterior_belief.mean(0)
        mask = posterior_belief >= thresholds
        p_star = mask.sum(0) / mask.shape[0]
        if_star = (posterior_belief * mask).sum(0) / (mask.sum(0) + 1e-12)
        variances = p_star * (if_star - expected_rewards) ** 2
        variances = np.maximum(variances, 1e-12)

        x = cp.Variable(self.n_items, pos=True)
        rewards = cp.Parameter(self.n_items,)
        gains = cp.Parameter(self.n_items, pos=True)
        deltas = r_star - x @ rewards
        exp_gain = x @ gains
        information_ratio = cp.quad_over_lin(deltas, exp_gain)
        objective = cp.Minimize(information_ratio)
        constraints = [0 <= x, x <= 1, cp.sum(x) == self.subset_size]
        prob = cp.Problem(objective, constraints,)
        rewards.value = expected_rewards
        gains.value = variances

        try:
            prob.solve(solver="ECOS")
            zeros_index = x.value < 1e-3
            ones_index = x.value > 1 - 1e-3
            nzeros = zeros_index.sum()
            nones = ones_index.sum()
            nitems = x.value.shape[0]
            logging.debug(
                f"{nitems - nones - nzeros} nstrict, {nones} ones, {nzeros} zeroes, {nitems} total items"
            )
            if (nitems - nones - nzeros) == 2:
                all_items = np.arange(nitems)
                strict_items = all_items[
                    ~np.bitwise_or(zeros_index, ones_index)
                ]
                probas = x.value[~np.bitwise_or(zeros_index, ones_index)]
                assert strict_items.shape[0] == 2, strict_items
                assert probas.shape[0] == 2, probas
                # 2 items to randomize the selection over
                logging.debug(f"items: {strict_items}, with probas: {probas}",)
                rho = probas[0]
                u = np.random.rand()
                if rho <= u:
                    remaning_item = strict_items[0]
                else:
                    remaning_item = strict_items[1]
                action = np.sort(
                    np.concatenate(
                        [
                            act_optimally(
                                x.value, top_k=self.subset_size - 1
                            ),
                            np.array([remaning_item]),
                        ]
                    )
                )
            else:
                action = act_optimally(x.value, top_k=self.subset_size)
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
                logging.debug(f"obj{prob.value}")
        except cp.SolverError:
            logging.warning("solver error")
            posterior_belief = self.sample_from_posterior(1)
            action = act_optimally(
                np.squeeze(posterior_belief), top_k=self.subset_size
            )
        except TypeError:
            logging.warning("solver error")
            posterior_belief = self.sample_from_posterior(1)
            action = act_optimally(
                np.squeeze(posterior_belief), top_k=self.subset_size
            )

        self.current_action = action
        self.c += 1
        return action

    def update_posterior(self, item_selected):
        try:
            item_selected = item_selected[0]
        except TypeError:
            assert(isinstance(item_selected, int))
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
