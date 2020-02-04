from mcmc import sample_from_posterior
from scipy.stats import uniform
from utils import act_optimally, possible_actions, expected_reward, optimized_ratio
from base_agents import Agent, EpochSamplingAgent
from collections import defaultdict
import numpy as np
from functools import partial

N_SAMPLES_V_IDS = 2


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
    probas_given_action = sampled_preferences[:, action]
    probas_given_action = probas_given_action / (1 + np.expand_dims(probas_given_action.sum(1), axis=-1))
    no_pick_given_action = 1 - probas_given_action.sum(1)
    p_no_item_action = no_pick_given_action.mean()
    probs += p_no_item_action
    for action_star, (p_star, theta_indices) in opt_actions.items():
        p_no_item_a_star_action = np.mean([no_pick_given_action[theta_indice] for theta_indice in theta_indices])
        g_a += p_no_item_a_star_action * np.log(p_no_item_a_star_action / (p_star * p_no_item_action))

    for action_ix, item_ix in enumerate(action):
        p_item_action = probas_given_action[:, action_ix].mean()
        probs += p_item_action
        for action_star, (p_star, theta_indices) in opt_actions.items():
            p_item_a_star_action = np.mean(
                [probas_given_action[theta_indice, action_ix] for theta_indice in theta_indices])
            g_a += p_item_a_star_action * np.log(p_item_a_star_action / (p_star * p_item_action))
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


def approximate_ids_action_selection(n, k, delta_, v_):
    v_information_ratios_items = - np.array([delta_([i]) ** 2 / v_([i]) for i in range(n)])
    if np.isinf(v_information_ratios_items.min()):
        # print('yo')
        # import ipdb;
        # ipdb.set_trace()
        v_information_ratios_items = - np.array([delta_([i]) for i in range(n)])
        return np.sort(np.argpartition(v_information_ratios_items, -k)[-k:])
    else:
        # print('yo')
        # import ipdb;
        # ipdb.set_trace()
        return np.sort(np.argpartition(v_information_ratios_items, -k)[-k:])


def ids_action_selection(n, k, delta_, g_):
    actions_set = possible_actions(n_items=n, assortment_size=k)
    min_information_ratio = np.inf
    ids_action = actions_set[0]
    for action1 in actions_set:
        for action2 in actions_set:
            g_a1, g_a2 = g_(action1), g_(action2)
            if (not g_a1) or (not g_a2):
                delta_1, delta_2 = delta_(action1), delta_(action2)
                if delta_1 < delta_2:
                    value = delta_1
                    action_picked = action1
                else:
                    value = delta_2
                    action_picked = action2
            else:
                value, rho = optimized_ratio(d1=delta_(action1),
                                             d2=delta_(action2),
                                             g1=g_a1,
                                             g2=g_a2)

                action_picked = action1 if np.random.rand() <= rho else action2
            if value < min_information_ratio:
                min_information_ratio = value
                ids_action = action_picked

    return ids_action


# TODO check out IDS with 16 and see the perf
# TODO longer horizon for mcmc experiments
# TODO check if the information ratio is bounded during simulations
# TODO find a faster package for multinomial logistic regression / alternative to pymc3
# TODO check out ICLR paper for approximating posterior distributions
# TODO approx Bayes method for multinomial dis
# TODO new experiments section in the Overleaf
class InformationDirectedSamplingAgent(Agent):
    def __init__(self, k, n, m=4):
        """
        :param k: assortment size
        :param n: number of items available
        :param m: number of posterior samples for IDS
        """
        super().__init__(k, n)
        self.prior_belief = uniform.rvs(size=(m, n))
        self.n_samples = m
        self.assortments_given = []
        self.item_picks = []
        self.optimal_actions = None
        self.g_ = None
        self.compute_g()
        self.r_star = 0.
        self.delta_ = None
        self.compute_delta()

    def update_r_star(self):
        sorted_beliefs = np.sort(self.prior_belief, axis=1)[:, -self.n_items:]  # shape (m, k)
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
        """
        :return:
        """
        self.update_r_star()
        self.delta_ = partial(delta_full,
                              sampled_preferences=self.prior_belief,
                              r_star=self.r_star)

    def compute_g(self):
        self.update_optimal_actions()
        self.g_ = partial(g_full,
                          sampled_preferences=self.prior_belief,
                          opt_actions=self.optimal_actions)

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

    def reset(self):
        self.prior_belief = uniform.rvs(size=(self.n_samples, self.n_items))
        self.assortments_given = []
        self.item_picks = []
        self.optimal_actions = None
        self.g_ = None
        self.compute_g()
        self.r_star = 0.
        self.delta_ = None
        self.compute_delta()
        # TODO make it cleaner

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        self.item_picks.append(item_selected)
        self.prior_belief = sample_from_posterior(n_samples=self.n_samples,
                                                  assortments=np.array(self.assortments_given),
                                                  item_picks=np.array(self.item_picks),
                                                  n_observations=len(self.item_picks),
                                                  n_items=self.n_items)
        self.compute_delta()
        self.compute_g()
        return reward


class ApproximateInformationDirectedSamplingAgent(EpochSamplingAgent):
    def __init__(self, k, n):
        super().__init__(k, n)
        self.optimal_actions = None
        self.v_ = None
        self.r_star = 0.
        self.delta_ = None
        self.n_samples = N_SAMPLES_V_IDS
        self.prior_belief = self.sample_from_posterior(self.n_samples)

    def update_r_star(self):
        sorted_beliefs = np.sort(self.prior_belief, axis=1)[:, -self.n_items:]  # shape (m, k)
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
        """
        :return:
        """
        self.update_r_star()
        self.delta_ = partial(delta_full,
                              sampled_preferences=self.prior_belief,
                              r_star=self.r_star)

    def compute_v(self):
        self.v_ = partial(g_full,
                          sampled_preferences=self.prior_belief,
                          opt_actions=self.optimal_actions)

    def proposal(self):
        self.prior_belief = self.sample_from_posterior(self.n_samples)
        self.update_r_star()
        self.compute_delta()
        self.update_optimal_actions()
        self.compute_v()
        action = np.array(ids_action_selection(n=self.n_items,
                                               k=self.assortment_size,
                                               delta_=self.delta_,
                                               g_=self.v_))
        # print(self.posterior_parameters)
        # print(self.prior_belief)
        # print(np.array([self.delta_([i]) ** 2 / self.v_([i]) for i in range(self.n_items)]))
        # import ipdb;
        # ipdb.set_trace()
        self.current_action = action
        return action

    def reset(self):
        self.epoch_ended = True
        self.current_action = self.n_items
        self.epoch_picks = defaultdict(int)
        self.posterior_parameters = [(1, 1) for _ in range(self.n_items)]
        self.optimal_actions = None
        self.v_ = None
        self.r_star = 0.
        self.delta_ = None
        self.n_samples = N_SAMPLES_V_IDS
        self.prior_belief = self.sample_from_posterior(self.n_samples)

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        if item_selected == self.n_items:
            self.epoch_ended = True
            n_is = [int(ix in self.current_action) for ix in range(self.n_items)]
            v_is = [self.epoch_picks[i] for i in range(self.n_items)]
            self.posterior_parameters = [(a + n_is[ix], b + v_is[ix]) for ix, (a, b) in
                                         enumerate(self.posterior_parameters)]
            self.epoch_picks = defaultdict(int)
        else:
            self.epoch_picks[item_selected] += 1
            self.epoch_ended = False
        return reward


# class VarianceBasedIDSAgent(Agent):
#     def __init__(self, k, n, m=4):
#         """
#         :param k: assortment size
#         :param n: number of items available
#         :param m: number of posterior samples for IDS
#         """
#         super().__init__(k, n)
#         self.n_samples = m  # new parameter = number of posterior samples drawn
#         self.assortments_given = []
#         self.item_picks = []
#         # sampling of the m priors
#         self.prior_belief = uniform.rvs(size=(m, n))
#         # maintaining information about the corresponding optimal actions
#         self.optimal_actions_information = self.from_beliefs_to_actions_information()
#
#     def act(self):
#         """
#         3 steps:
#         - loop over M posterior samples and:
#             get p(a*)
#         - loop over items + no_items and:
#             compu
#         :return:
#         """
#         action = act_optimally(self.prior_beliefs, top_k=self.assortment_size)
#         assortment = np.zeros(self.n_items + 1)
#         assortment[self.n_items] = 1.
#         for item in action:
#             assortment[item] = 1.
#         self.assortments_given.append(assortment)
#         return action
#
#     def reset(self):
#         self.prior_beliefs = uniform.rvs(size=self.n_items)
#         self.assortments_given = []
#         self.item_picks = []
#
#     def update(self, item_selected):
#         reward = self.perceive_reward(item_selected)
#         self.item_picks.append(item_selected)
#         self.prior_beliefs = np.squeeze(sample_from_posterior(n_samples=1,
#                                                               assortments=np.array(self.assortments_given),
#                                                               item_picks=np.array(self.item_picks),
#                                                               n_observations=len(self.item_picks),
#                                                               n_items=self.n_items))
#         return reward
#
#     def from_beliefs_to_actions_information(self):
#         """
#         :return: dictionary of informations about optimal action for each posterior sample of the model parameters
#         # keys: actions = sorted tuple of items to propose in the assortment
#         # values: (p(action = a*), [thetas such that action is optimal for theta]
#         """
#         posteriors_actions = [tuple(act_optimally(belief=posterior_sample)) for posterior_sample in self.prior_beliefs]
#         optimal_actions_information = defaultdict(list)
#         for ix, action in enumerate(posteriors_actions):
#             optimal_actions_information[action].append(ix)
#
#         return {action: (len(theta_idxs) / self.n_samples, theta_idxs) for
#                 action, theta_idxs in self.optimal_actions_information.items()}


if __name__ == "__main__":
    pref = np.random.rand(2, 4)
    opt_act = {(0, 1): [1., [0, 1]]}
    actions = possible_actions(4, 2)
    for action in actions:
        print(g_full(action, pref, opt_act))
