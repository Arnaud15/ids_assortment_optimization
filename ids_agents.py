from cids_utils import expected_regrets, solve_cvx, var_if_a_star, kl_if_a_star
from ids_utils import (
    InformationDirectedSampler,
    ids_exact_action,
    information_ratio,
    info_gain_step,
    delta_step,
)
from env import act_optimally, possible_actions
from ts_agents import EpochSamplingTS
import numpy as np
import logging


class EpochSamplingCIDS(EpochSamplingTS):
    def __init__(
        self, k, n, n_samples, info_type, **kwargs,
    ):
        EpochSamplingTS.__init__(
            self, k, n, sampling=False,
        )
        self.n_samples = n_samples
        assert info_type in {"variance", "gain"}
        self.info_type = info_type

    def proposal(self):
        posterior_belief = self.sample_from_posterior(self.n_samples)
        regrets = expected_regrets(
            posterior_belief=posterior_belief, assortment_size=self.subset_size
        )

        variances = var_if_a_star(
            posterior_belief=posterior_belief, assortment_size=self.subset_size
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
