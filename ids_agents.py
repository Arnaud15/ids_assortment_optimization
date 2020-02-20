import abc
import mcmc
from scipy.stats import uniform
from utils import act_optimally, possible_actions, expected_reward, optimized_ratio
from base_agents import Agent, EpochSamplingAgent, HypermodelAgent
from collections import defaultdict
import numpy as np
from functools import partial
from random import shuffle

DISCRETE_IDS_OPTIMIZATION = True


def delta_full(action, sampled_preferences, r_star):
    return r_star - expected_reward(action=action, preferences=sampled_preferences)

def g_full(action, sampled_preferences, opt_actions):
    """
    :param action:
    :param sampled_preferences: sampled posterior thetas
    :param opt_actions: dictionary {action_tuple:p_action, theta_indices}
    :return:
    """
    g_a = 0.
    probs = 0.
    M = sampled_preferences.shape[0]
    probas_given_action = sampled_preferences[:, action]
    probas_given_action = probas_given_action / (1 + np.expand_dims(probas_given_action.sum(1), axis=-1))
    no_pick_given_action = 1 - probas_given_action.sum(1)
    p_no_item_action = no_pick_given_action.mean()
    probs += p_no_item_action
    for action_star, (p_star, theta_indices) in opt_actions.items():
        p_no_item_a_star_action = np.sum([no_pick_given_action[theta_indice] for theta_indice in theta_indices]) / M
        g_a += p_no_item_a_star_action * np.log(p_no_item_a_star_action / (p_star * p_no_item_action))

    for action_ix, item_ix in enumerate(action):
        p_item_action = probas_given_action[:, action_ix].mean()
        if p_item_action:
            probs += p_item_action
            for action_star, (p_star, theta_indices) in opt_actions.items():
                p_item_a_star_action = np.sum(
                    [probas_given_action[theta_indice, action_ix] for theta_indice in theta_indices]) / M
                if p_item_a_star_action:
                    g_a += p_item_a_star_action * np.log(p_item_a_star_action / (p_star * p_item_action))
    assert probs > 0.999, f"{probs}"
    assert probs < 1.001, f"{probs}"
    return g_a


def v_full(action, sampled_preferences, opt_actions):
    if len(opt_actions.keys()) > 1:
        r_a_t_given_a_star = np.array(
            [expected_reward(sampled_preferences[thetas_a_star, :], action) for a_star, (p_a_star, thetas_a_star) in
             opt_actions.items()])
        probas_a_star = np.array([p_a_star for a_star, (p_a_star, thetas_a_star) in opt_actions.items()])
        return probas_a_star.dot(r_a_t_given_a_star ** 2) - (probas_a_star.dot(r_a_t_given_a_star)) ** 2
    else:
        return 0.


# TODO fix with greedy
def approximate_ids_action_selection(n, k, delta_, v_):
    v_information_ratios_items = - np.array([delta_([i]) ** 2 / v_([i]) for i in range(n)])
    if np.isinf(v_information_ratios_items.min()):
        v_information_ratios_items = - np.array([delta_([i]) for i in range(n)])
        return np.sort(np.argpartition(v_information_ratios_items, -k)[-k:])
    else:
        return np.sort(np.argpartition(v_information_ratios_items, -k)[-k:])


def ids_action_selection(n, k, delta_, g_):
    actions_set = possible_actions(n_items=n, assortment_size=k)
    shuffle(actions_set)
    min_information_ratio = np.inf
    rho_pick = None
    deltas = [None, None]
    gains = [None, None]
    ids_action = actions_set[0]
    total_no_info_gain = 0
    total_no_delta = 0
    for action1 in actions_set:
        g_a1 = g_(action1)
        delta_1 = delta_(action1)
        if not g_a1:
            total_no_info_gain += 1
        if not delta_1:
            total_no_delta += 1
        for action2 in actions_set:
            g_a2 = g_(action2)
            delta_2 = delta_(action2)
            if (not g_a1) or (not g_a2):
                if delta_1 < delta_2:
                    value = delta_1
                    action_picked = action1
                else:
                    value = delta_2
                    action_picked = action2
                rho = 1. if delta_1 < delta_2 else 0.
            else:
                value, rho = optimized_ratio(d1=delta_1,
                                             d2=delta_2,
                                             g1=g_a1,
                                             g2=g_a2,
                                             discrete=DISCRETE_IDS_OPTIMIZATION)

                action_picked = action1 if np.random.rand() <= rho else action2
            if value < min_information_ratio:
                deltas = delta_1, delta_2
                gains = g_a1, g_a2
                min_information_ratio = value
                ids_action = action_picked
                rho_pick = rho

    # print(f"min information ratio obtained is {min_information_ratio:.4f}")
    # print(f"with deltas: {[f'{delt:.5f}' for delt in deltas]}")
    # print(f"and information gains: {[f'{gain:.5f}' for gain in gains]}")
    # print(f"And rho = {rho_pick}")
    # print(f"with total no info gain share = {total_no_info_gain / len(actions_set):.2f}")
    # print(f"with total no delta = {total_no_delta}")
    # if total_no_info_gain:
    #     import ipdb
    #     ipdb.set_trace()
    # if total_no_delta:
    #     import ipdb
    #     ipdb.set_trace()
    return ids_action
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
