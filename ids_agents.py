from ids_utils import InformationDirectedSampler, ids_action_selection_numba
from base_agents import EpochSamplingAgent, HypermodelAgent
import numpy as np
from utils import possible_actions


class EpochSamplingIDS(EpochSamplingAgent):
    def __init__(self, k, n, correlated_sampling, n_samples, info_type, **kwargs):
        EpochSamplingAgent.__init__(self, k, n, horizon=None, correlated_sampling=False)
        self.ids_sampler = InformationDirectedSampler(assortment_size=k, info_type=info_type, n_samples=n_samples)
        self.all_actions = np.array(possible_actions(self.n_items, self.assortment_size), dtype=int)

    def proposal(self):
        self.prior_belief = self.sample_from_posterior(self.ids_sampler.n_samples)
        # print(f"belief sampled is: {1000 * self.prior_belief.astype(int)}")
        # print(f"optimal actions are: {self.optimal_actions}")
        self.ids_sampler.update_belief(self.prior_belief)
        action = np.array(ids_action_selection_numba(g_=self.ids_sampler.g_,
                                               actions_set=self.all_actions,
                                               sampled_preferences=self.prior_belief,
                                               r_star=self.ids_sampler.r_star,
                                               actions_star=self.ids_sampler.actions_star,
                                               counts_star=self.ids_sampler.counts_star,
                                               thetas_star=self.ids_sampler.thetas_star))
        self.current_action = action
        # print("-" * 15)
        return action


class HypermodelIDS(HypermodelAgent):
    def __init__(self, k, n, n_samples, params, info_type, **kwargs):
        HypermodelAgent.__init__(self, k, n, params, n_samples=n_samples)
        self.ids_sampler = InformationDirectedSampler(assortment_size=k, info_type=info_type, n_samples=n_samples)
        self.all_actions = np.array(possible_actions(self.n_items, self.assortment_size), dtype=int)

    def act(self):
        self.prior_belief = self.sample_from_posterior(self.n_samples)
        # print(f"belief sampled is: {1000 * self.prior_belief.astype(int)}")
        # print(f"optimal actions are: {self.optimal_actions}")
        self.ids_sampler.update_belief(self.prior_belief)
        action = np.array(ids_action_selection_numba(g_=self.ids_sampler.g_,
                                               actions_set=self.all_actions,
                                               sampled_preferences=self.prior_belief,
                                               r_star=self.ids_sampler.r_star,
                                               actions_star=self.ids_sampler.actions_star,
                                               counts_star=self.ids_sampler.counts_star,
                                               thetas_star=self.ids_sampler.thetas_star))
        self.current_action = action
        # print("-" * 15)
        return action

# TODO put back together the MCMC part
# class InformationDirectedSampling(abc.ABC):
#     def __init__(self, number_of_ids_samples, information_type='ids'):