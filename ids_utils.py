import numba
import numpy as np
from random import shuffle
from collections import defaultdict
from functools import partial
from utils import act_optimally, possible_actions, expected_reward, optimized_ratio


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

def ids_action_selection(n, k, delta_, g_):
    actions_set = possible_actions(n_items=n, assortment_size=k)
    shuffle(actions_set)
    min_information_ratio = np.inf
    # rho_pick = 0.5
    # deltas = [None, None]
    # gains = [None, None]
    ids_action = actions_set[0]
    # total_no_info_gain = 0
    # total_no_delta = 0
    for action1 in actions_set:
        g_a1 = g_(action1)
        delta_1 = delta_(action1)
        # if not g_a1:
        #     total_no_info_gain += 1
        # if not delta_1:
        #     total_no_delta += 1
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
                # deltas = delta_1, delta_2
                # gains = g_a1, g_a2
                min_information_ratio = value
                ids_action = action_picked
                # rho_pick = rho
    return ids_action
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
    # return ids_action


class InformationDirectedSampler:
    def __init__(self, assortment_size, n_samples, info_type):
        """
        info_type == IDS for information ratio or VIDS or variance-based information ratio
        """
        self.assortment_size = assortment_size
        self.n_samples = n_samples
        self.info_type = info_type
        self.init_sampler()

    def init_sampler(self):
        self.posterior_belief = np.random.rand(self.n_samples, self.assortment_size)
        self.optimal_actions = None
        self.optimal_actions = None
        self.thetas_nb = None
        self.counts_nb = None
        self.g_ = None
        self.r_star = 0.
        self.delta_ = None

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
        self.opt_actions_nb = np.array([list(key) for key in optimal_actions_information.keys()])
        self.counts_nb = np.array([len(val) for val in optimal_actions_information.values()])
        self.thetas_nb = []
        for val in optimal_actions_information.values():
            self.thetas_nb += val
        self.thetas_nb = np.array(self.thetas_nb)


    def compute_delta(self):
        self.update_r_star()
        self.delta_ = delta_full
                            #   sampled_preferences=self.posterior_belief,
                            #   r_star=self.r_star)

    def compute_g(self):
        if self.info_type == "IDS":
            self.g_ = g_full
                            # sampled_preferences=self.posterior_belief,
                            # opt_actions=self.optimal_actions)
        elif self.info_type =="VIDS":
            self.g_ = v_full
                            # sampled_preferences=self.posterior_belief,
                            # opt_actions=self.optimal_actions)
        else:
            raise ValueError

    def update_belief(self, new_belief):
        self.posterior_belief = new_belief
        self.compute_delta()
        self.update_optimal_actions()
        self.compute_g()


#TODO: check VIDS correctness
# TODO fix with greedy
# def approximate_ids_action_selection(n, k, delta_, v_):
#     v_information_ratios_items = - np.array([delta_([i]) ** 2 / v_([i]) for i in range(n)])
#     if np.isinf(v_information_ratios_items.min()):
#         v_information_ratios_items = - np.array([delta_([i]) for i in range(n)])
#         return np.sort(np.argpartition(v_information_ratios_items, -k)[-k:])
#     else:
#         return np.sort(np.argpartition(v_information_ratios_items, -k)[-k:])