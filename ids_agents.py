from cvxpy.expressions.cvxtypes import pos
from cids_utils import (
    expected_regrets,
    solve_cvx,
    var_if_a_star,
    kl_if_a_star,
    kl_ids,
)
from ids_utils import (
    InformationDirectedSampler,
    ids_exact_action,
    information_ratio,
    info_gain_step,
    delta_step,
)
from jax_utils import *
from base_agents import x_beta_sampling
from env import act_optimally, possible_actions
from ts_agents import EpochSamplingTS
import numpy as np
import scipy.stats as sts
import jax.numpy as jnp
import logging


THETAS = sts.norm.isf(np.linspace(0.999, 0.001, num=9))


class EpochSamplingCorrIDS(EpochSamplingTS):
    def __init__(
        self, k, n, n_samples, info_type, **kwargs,
    ):
        EpochSamplingTS.__init__(
            self, k, n, sampling=False,
        )
        self.n_samples = 5
        self.hasher = self.n_items ** jnp.arange(self.subset_size)
        self.get_top_actions = top_actions_factory(self.subset_size)
        self.hash_actions = hash_actions_factory(
            hasher=self.n_items ** jnp.arange(self.subset_size)
        )
        self.expected_rewards = all_er()
        self.flat_rewards = flat_er()
        self.get_variances = variances_factory()

    def strict_ts_cs_actions(self):
        correlated_sample = x_beta_sampling(
            a_s=self._n_is,
            b_s=self._v_is,
            correlated_sampling=True,
            n_samples=0,
            input_thetas=THETAS,
        )
        correlated_actions = self.get_top_actions(correlated_sample)
        hashed_correlated_actions = self.hash_actions(correlated_actions)
        _, ixs_corr = jnp.unique(hashed_correlated_actions, True)
        logging.info(
            f"{THETAS.shape[0]} thetas, {ixs_corr.shape[0]} actual correlated assortments"
        )
        return correlated_actions[ixs_corr, :]

    def proposal(self):
        ts_cs_actions = self.strict_ts_cs_actions()

        posterior_belief = self.sample_from_posterior(self.n_samples)

        expected_rewards_all = self.expected_rewards(
            posterior_belief, ts_cs_actions
        )
        means = jnp.mean(expected_rewards_all, axis=1)
        top_actions = self.get_top_actions(posterior_belief)
        best_expected_reward_per_sample = self.flat_rewards(
            posterior_belief, top_actions
        )
        r_star = jnp.mean(best_expected_reward_per_sample)
        logging.info(f"r_star: {r_star:.2f}")
        regrets = r_star - means

        hashed_top_actions = self.hash_actions(top_actions)
        _, unique_ixs, row_to_uix, counts = jnp.unique(
            hashed_top_actions, True, True, True
        )
        logging.info(f"{unique_ixs.shape[0]} distinct top actions")
        variances = self.get_variances(
            row_to_uix,
            unique_ixs.shape[0],
            counts,
            counts / counts.sum(),
            expected_rewards_all,
            means,
        )
        variances = jnp.maximum(variances, 1e-12)

        action_ix = solve_mixture_jax(regrets=regrets, variances=variances,)
        action = np.array(ts_cs_actions[action_ix])

        self.current_action = action
        return action


class EpochSamplingThompsonIDS(EpochSamplingTS):
    def __init__(
        self, k, n, n_samples, info_type, **kwargs,
    ):
        EpochSamplingTS.__init__(
            self, k, n, sampling=False,
        )
        self.n_samples = 200
        assert info_type in {"variance", "gain"}
        self.info_type = info_type

    def proposal(self):
        posterior_belief = self.sample_from_posterior(self.n_samples)
        regrets = expected_regrets(
            posterior_belief=posterior_belief, assortment_size=self.subset_size
        )
        if self.info_type == "gain":
            variances = kl_ids(
                posterior_belief=posterior_belief, subset_size=self.subset_size
            )
        else:
            variances = var_if_a_star(
                posterior_belief=posterior_belief,
                assortment_size=self.subset_size,
            )

        action = solve_cvx(
            regrets=regrets,
            variances=variances,
            subset_size=self.subset_size,
            n_items=self.n_items,
        )

        self.current_action = action
        return action


class EpochSamplingIDS(EpochSamplingTS):
    def __init__(
        self, k, n, n_samples, info_type, **kwargs,
    ):
        EpochSamplingTS.__init__(
            self, k, n, sampling=False,
        )
        self.ids_sampler = InformationDirectedSampler(
            n_items=n,
            assortment_size=k,
            info_type=info_type,
            n_samples=n_samples,
            dynamics="step",
        )
        self.all_actions = np.array(
            possible_actions(self.n_items, self.subset_size), dtype=int,
        )

    def proposal(self):
        self.prior_belief = self.sample_from_posterior(
            self.ids_sampler.n_samples
        )
        self.ids_sampler.update_belief(self.prior_belief)
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
        self.current_action = assortment
        self.data_stored["info_ratio"].append(ir_assortment)
        self.data_stored["entropy_a_star"].append(
            self.ids_sampler.a_star_entropy
        )
        self.data_stored["rho_policy"].append(rho_policy)
        self.data_stored["delta_min_2"].append(self.ids_sampler.delta_min ** 2)
        return assortment
