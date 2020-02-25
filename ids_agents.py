from ids_utils import InformationDirectedSampler, ids_action_selection
from base_agents import EpochSamplingAgent, HypermodelAgent
import numpy as np


class EpochSamplingIDS(EpochSamplingAgent):
    def __init__(self, k, n, correlated_sampling, n_samples, info_type, **kwargs):
        EpochSamplingAgent.__init__(self, k, n, horizon=None, correlated_sampling=False)
        self.ids_sampler = InformationDirectedSampler(assortment_size=k, info_type=info_type, n_samples=n_samples)

    def proposal(self):
        self.prior_belief = self.sample_from_posterior(self.n_samples)
        # print(f"belief sampled is: {1000 * self.prior_belief.astype(int)}")
        # print(f"optimal actions are: {self.optimal_actions}")
        self.ids_sampler.update_belief(self.prior_belief)
        action = np.array(ids_action_selection(n=self.n_items,
                                               k=self.assortment_size,
                                               delta_=self.ids_sampler.delta_,
                                               g_=self.ids_sampler.g_))
        self.current_action = action
        # print("-" * 15)
        return action


class HypermodelIDS(HypermodelAgent):
    def __init__(self, k, n, n_samples, params, info_type, **kwargs):
        HypermodelAgent.__init__(self, k, n, params, n_samples=n_samples)
        self.ids_sampler = InformationDirectedSampler(assortment_size=k, info_type=info_type, n_samples=n_samples)

    def act(self):
        self.prior_belief = self.sample_from_posterior(self.n_samples)
        # print(f"belief sampled is: {1000 * self.prior_belief.astype(int)}")
        # print(f"optimal actions are: {self.optimal_actions}")
        self.ids_sampler.update_belief(self.prior_belief)
        action = np.array(ids_action_selection(n=self.n_items,
                                               k=self.assortment_size,
                                               delta_=self.ids_sampler.delta_,
                                               g_=self.ids_sampler.g_))
        self.current_action = action
        # print("-" * 15)
        return action

# TODO put back together the MCMC part
# class InformationDirectedSampling(abc.ABC):
#     def __init__(self, number_of_ids_samples, information_type='ids'):
# TODO remember the greedy algo for action selection