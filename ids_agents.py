import abc
import mcmc
from scipy.stats import uniform
from utils import act_optimally, possible_actions, expected_reward, optimized_ratio
from base_agents import Agent, EpochSamplingAgent, HypermodelAgent
from collections import defaultdict
import numpy as np
from functools import partial
from random import shuffle



class InformationDirectedSampling(abc.ABC):
    def __init__(self, number_of_ids_samples, information_type='ids'):
        self.n_samples = number_of_ids_samples
        self.info_type = information_type
        self.init_posterior_representation()
        self.optimal_actions = None
        self.g_ = None
        self.r_star = 0.
        self.delta_ = None
        self.posterior_belief = self.sample_from_posterior(self.n_samples) 

    @abc.abstractmethod
    def init_posterior_representation(self):
        pass

    @abc.abstractmethod
    def sample_from_posterior(self, num_samples):
        pass

    def update_r_star(self):
        sorted_beliefs = np.sort(self.posterior_belief, axis=1)[:, -self.assortment_size:]  # shape (m, k)
        picking_probabilities = sorted_beliefs.sum(1)
        self.r_star = (picking_probabilities / (1 + picking_probabilities)).mean()

    def update_optimal_actions(self):
        """
        :return: dictionary of informations about optimal action for each posterior sample of the model parameters
        # keys: actions = sorted tuple of items to propose in the assortment
        # values: (p(action = a*), [thetas such that action is optimal for theta]
        """
        posteriors_actions = act_optimally(self.posterior_belief, self.assortment_size)
        posteriors_actions = [tuple(posteriors_actions[ix, :]) for ix in range(self.n_samples)]
        optimal_actions_information = defaultdict(list)
        for ix, action in enumerate(posteriors_actions):
            optimal_actions_information[action].append(ix)

        self.optimal_actions = {action: (len(theta_idxs) / self.n_samples, theta_idxs) for
                                action, theta_idxs in optimal_actions_information.items()}

    def compute_delta(self):
        self.update_r_star()
        self.delta_ = partial(delta_full,
                              sampled_preferences=self.posterior_belief,
                              r_star=self.r_star)

    def compute_g(self):
        if self.info_type == "ids":
            self.g_ = partial(g_full,
                            sampled_preferences=self.posterior_belief,
                            opt_actions=self.optimal_actions)
        elif self.info_type == "vids":
            self.g_ = partial(v_full,
                            sampled_preferences=self.posterior_belief,
                            opt_actions=self.optimal_actions)
        else:
            raise ValueError

    def reset(self):
        self.init_posterior_representation()
        self.optimal_actions = None
        self.g_ = None
        self.r_star = 0.
        self.delta_ = None
        self.posterior_belief = self.sample_from_posterior(self.n_samples)
    
# TODO new experiments section in the Overleaf
class InformationDirectedSamplingAgent(InformationDirectedSampling, Agent):
    def __init__(self, k, n, number_of_ids_samples, **kwargs):
        """
        :param k: assortment size
        :param n: number of items available
        :param n_ids_samples: number of posterior samples for IDS
        """
        Agent.__init__(self, k, n)
        InformationDirectedSampling.__init__(self, number_of_ids_samples)

    def init_posterior_representation(self):
        self.assortments_given = []
        self.item_picks = []

    def sample_from_posterior(self, n_samples):
        if not len(self.assortments_given):
            return uniform.rvs(size=(n_samples, self.n_items))
        else:
            return mcmc.sample_from_posterior(n_samples=n_samples,
                                                  assortments=np.array(self.assortments_given),
                                                  item_picks=np.array(self.item_picks),
                                                  n_observations=len(self.item_picks),
                                                  n_items=self.n_items)

    def act(self):
        """
        3 steps:
        - loop over M posterior samples and:
            get p(a*)
        - loop over items + no_items and:
            compu
        :return:
        """
        action = ids_action_selection(n=self.n_items,
                                      k=self.assortment_size,
                                      delta_=self.delta_,
                                      g_=self.g_)
        assortment = np.zeros(self.n_items + 1)
        assortment[self.n_items] = 1.
        for item in action:
            assortment[item] = 1.
        self.assortments_given.append(assortment)
        return np.array(action)

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        self.item_picks.append(item_selected)
        self.prior_belief = self.sample_from_posterior(self.n_samples)
        self.update_optimal_actions()
        self.compute_delta()
        self.compute_g()
        return reward


class EpochSamplingIDS(EpochSamplingAgent, InformationDirectedSampling):
    def __init__(self, k, n, horizon, number_of_ids_samples, correlated_sampling, **kwargs):
        EpochSamplingAgent.__init__(self, k, n, horizon=horizon, correlated_sampling=correlated_sampling)
        InformationDirectedSampling.__init__(self, number_of_ids_samples)

    def init_posterior_representation(self):
        self.current_action = self.n_items
        self.epoch_ended = True
        self.epoch_picks = defaultdict(int)
        self.posterior_parameters = [(1, 1) for _ in range(self.n_items)]

    def proposal(self):
        self.prior_belief = self.sample_from_posterior(self.n_samples)
        # print(f"belief sampled is: {1000 * self.prior_belief.astype(int)}")
        self.update_r_star()
        self.compute_delta()
        self.update_optimal_actions()
        # print(f"optimal actions are: {self.optimal_actions}")
        self.compute_g()
        action = np.array(ids_action_selection(n=self.n_items,
                                               k=self.assortment_size,
                                               delta_=self.delta_,
                                               g_=self.g_))
        self.current_action = action
        # print("-" * 15)
        return action

# TODO finish refactoring
class HypermodelDS(HypermodelAgent, InformationDirectedSampling):
    def __init__(self, k, n, num_samples, params, **kwargs):
        HypermodelAgent.__init__(self, k, n, params, n_samples=1)
        self.optimal_actions = None
        self.r_star = 1.
        self.g_ = None
        self.delta_ = None
        
    def update_r_star(self):
        sorted_beliefs = np.sort(self.prior_belief, axis=1)[:, -self.assortment_size:]  # shape (m, k)
        picking_probabilities = sorted_beliefs.sum(1)
        self.r_star = (picking_probabilities / (1 + picking_probabilities)).mean()

    def update_optimal_actions(self):
        """
        :return: dictionary of informations about optimal action for each posterior sample of the model parameters
        # keys: actions = sorted tuple of items to propose in the assortment
        # values: (p(action = a*), [thetas such that action is optimal for theta]
        """

        posteriors_actions = act_optimally(self.prior_belief, self.assortment_size)
        posteriors_actions = [tuple(posteriors_actions[ix, :]) for ix in range(self.n_samples)]
        optimal_actions_information = defaultdict(list)
        for ix, action in enumerate(posteriors_actions):
            optimal_actions_information[action].append(ix)

        self.optimal_actions = {action: (len(theta_idxs) / self.n_samples, theta_idxs) for
                                action, theta_idxs in optimal_actions_information.items()}

    def compute_delta(self):
        self.update_r_star()
        self.delta_ = partial(delta_full,
                              sampled_preferences=self.prior_belief,
                              r_star=self.r_star)

    def compute_g(self):
        self.g_ = partial(g_full,
                          sampled_preferences=self.prior_belief,
                          opt_actions=self.optimal_actions)

    def act(self):
        self.prior_belief = self.sample_from_posterior(self.n_samples)
        # print(f"belief sampled is: {1000 * self.prior_belief.astype(int)}")
        self.update_r_star()
        self.compute_delta()
        self.update_optimal_actions()
        # print(f"optimal actions are: {self.optimal_actions}")
        self.compute_g()
        action = np.array(ids_action_selection(n=self.n_items,
                                               k=self.assortment_size,
                                               delta_=self.delta_,
                                               g_=self.g_))
        # print("-" * 15)
        # action = np.random.choice(np.arange(self.n_items, dtype=int), size=self.assortment_size, replace=False)
        self.current_action = action
        return action


if __name__ == "__main__":
    pref = np.random.rand(2, 4)
    opt_act = {(0, 1): [1., [0, 1]]}
    actions = possible_actions(4, 2)
    for action in actions:
        print(g_full(action, pref, opt_act))
