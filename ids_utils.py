import numba
import numpy as np
from utils import act_optimally
from collections import defaultdict

DISCRETIZATION_IDS = 25
RHO_VALUES = np.linspace(start=0., stop=1., num=DISCRETIZATION_IDS)

@numba.jit(nopython=True)
def g_full_numba(action, sampled_preferences, actions_star, counts, thetas):
    """
    :param action: 1D array of size (K,) with the indices of the current action
    :param sampled_preferences: sampled posterior thetas of shape (M, N)
    :param actions_star: 2D array of shape (n_top_actions, assortment_size)
    :param counts: 1D array of shape (n_top_actions,) = how many theta for each opt action 
    :param thetas: 1D array of the indices in [0, M-1] of the thetas associated w/ each opt action
    :return:
    """
    g_a = 0.
    probs = 0.
    M = sampled_preferences.shape[0]
    K = action.shape[0]
    n_actions_star = actions_star.shape[0]
    
    probas_given_action = np.zeros((M, K))  # Probabilities of each item given action
    no_pick_given_action = np.zeros((M,)) # Same for no pick given action
    p_no_item_action = 0.
    for m in range(M):
        sum_row = 0.
        for k in range(K):
            val = sampled_preferences[m, action[k]]
            probas_given_action[m, k] = val
            sum_row += val
        probas_given_action[m, :] = probas_given_action[m, :] / (1 + sum_row)
        no_pick_given_action[m] = 1 - (sum_row / (1 + sum_row))
        p_no_item_action += no_pick_given_action[m]
    p_no_item_action = p_no_item_action / M

    probs += p_no_item_action
    theta_start = 0
    for i in range(n_actions_star): # First we treat separately the y=NO_ITEM case
        theta_indices = thetas[theta_start:theta_start+counts[i]]
        theta_start += counts[i]
        p_star = counts[i] / M
        p_no_item_a_star_action = 0.
        for theta_indice in theta_indices:
            p_no_item_a_star_action += no_pick_given_action[theta_indice]
        p_no_item_a_star_action = p_no_item_a_star_action / M
        g_a += p_no_item_a_star_action * np.log(p_no_item_a_star_action / (p_star * p_no_item_action))

    for ix in range(K): # Now other y s are examined
        p_item_action = 0.
        for m in range(M):
            p_item_action += probas_given_action[m, ix]
        p_item_action /= M
        if p_item_action > 1e-8:
            probs += p_item_action
            theta_start = 0
            for j in range(n_actions_star):
                p_star = counts[j] / M
                p_item_a_star_action = 0.
                theta_indices = thetas[theta_start:theta_start+counts[j]]
                for theta_indice in theta_indices:
                    p_item_a_star_action += probas_given_action[theta_indice, ix]
                theta_start += counts[j]
                p_item_a_star_action /= M
                if p_item_a_star_action:
                    g_a += p_item_a_star_action * np.log(p_item_a_star_action / (p_star * p_item_action))
    if (probs < 0.999) or (probs > 1.001):
        raise ValueError('Problem in IDS with probabilities not summing to 1')
    return g_a

@numba.jit(nopython=True)
def numba_expected_reward(pref, action):
    """
    param: pref 2D array shape (M, N) sampled preferences
    param: action 1D array of shape (K,) items selected in assortment
    return: expected reward for action = scalar
    """
    M = pref.shape[0]
    K = action.shape[0]
    result = 0.
    for i in range(M):
        temp_sum = 0.
        for k in range(K):
            temp_sum += pref[i, action[k]]
        result += temp_sum / (1 + temp_sum)
    return result / M

@numba.jit(nopython=True)
def delta_full_numba(action, sampled_preferences, r_star):
    """
    param: action 1D array of shape (K,) items selected in assortment
    param: preferences 2D array shape (M, N) sampled preferences
    param: r_star expected reward from taking optimal action for each theta model possible
    return: r_star - exp_reward
    """
    x = r_star - numba_expected_reward(action=action, pref=sampled_preferences)
    return x

@numba.jit(nopython=True)
def information_ratio_numba(rho, d1, d2, g1, g2):
    return (d1 * rho + (1 - rho) * d2) ** 2 / (g1 * rho + (1 - rho) * g2)

@numba.jit(nopython=True)
def optimized_ratio_numba(d1, d2, g1, g2):
    n_rho = RHO_VALUES.shape[0]
    min_ = 1e8
    rho_min = -1
    for ix in range(n_rho):
        rho = RHO_VALUES[ix]
        val = information_ratio_numba(rho, d1, d2, g1, g2)
        if val < min_:
            rho_min = rho
            min_ = val
    return min_, rho_min

@numba.jit(nopython=True)
def ids_action_selection_numba(g_, actions_set, sampled_preferences, r_star, actions_star, counts_star, thetas_star):
    """
    param: g_ = information_ratio computation function selected
    param: actions_set = possible actions given n, k
    param: preferences 2D array shape (M, N) sampled preferences
    param: r_star expected reward from taking optimal action for each theta model possible
    :param actions_star: 2D array of shape (n_top_actions, assortment_size)
    :param counts: 1D array of shape (n_top_actions,) = how many theta for each opt action 
    :param thetas: 1D array of the indices in [0, M-1] of the thetas associated w/ each opt action
    """
    # Shuffling the actions set
    np.random.shuffle(actions_set)
    # Quantities to keep track off
    min_information_ratio = 1e8
    ids_action = actions_set[0]
    n_actions = actions_set.shape[0]
    for i in range(n_actions):
        action1 = actions_set[i]
        g_a1 = g_(action1, sampled_preferences, actions_star, counts_star, thetas_star)
        delta_1 = delta_full_numba(action1, sampled_preferences, r_star)
        for j in range(n_actions):
            action2 = actions_set[j]
            g_a2 = g_(action2, sampled_preferences, actions_star, counts_star, thetas_star)
            delta_2 = delta_full_numba(action2, sampled_preferences, r_star)
            if (not g_a1) or (not g_a2):
                if delta_1 < delta_2:
                    value = delta_1
                    action_picked = action1
                else:
                    value = delta_2
                    action_picked = action2
                rho = 1. if delta_1 < delta_2 else 0.
            else:
                value, rho = optimized_ratio_numba(d1=delta_1,
                                                   d2=delta_2,
                                                   g1=g_a1,
                                                   g2=g_a2)
                action_picked = action1 if np.random.rand() <= rho else action2
            if value < min_information_ratio:
                min_information_ratio = value
                ids_action = action_picked
    return ids_action


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
        self.actions_star = None
        self.thetas_star = None
        self.counts_star = None
        if self.info_type == "IDS":
            self.g_ = g_full_numba
        else:
            raise ValueError('currently supported info types: IDS')
        self.r_star = 0.

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
        self.actions_star = np.array([list(key) for key in optimal_actions_information.keys()])
        self.counts_star = np.array([len(val) for val in optimal_actions_information.values()])
        self.thetas_star = []
        for val in optimal_actions_information.values():
            self.thetas_star += val
        self.thetas_star = np.array(self.thetas_star)

    def update_belief(self, new_belief):
        self.posterior_belief = new_belief
        self.update_r_star()
        self.update_optimal_actions()


#TODO: VIDS numba
#TODO: greedy information ratio
# def v_full(action, sampled_preferences, opt_actions):
#     if len(opt_actions.keys()) > 1:
#         r_a_t_given_a_star = np.array(
#             [expected_reward(sampled_preferences[thetas_a_star, :], action) for a_star, (p_a_star, thetas_a_star) in
#              opt_actions.items()])
#         probas_a_star = np.array([p_a_star for a_star, (p_a_star, thetas_a_star) in opt_actions.items()])
#         return probas_a_star.dot(r_a_t_given_a_star ** 2) - (probas_a_star.dot(r_a_t_given_a_star)) ** 2
#     else:
#         return 0


# def approximate_ids_action_selection(n, k, delta_, v_):
#     v_information_ratios_items = - np.array([delta_([i]) ** 2 / v_([i]) for i in range(n)])
#     if np.isinf(v_information_ratios_items.min()):
#         v_information_ratios_items = - np.array([delta_([i]) for i in range(n)])
#         return np.sort(np.argpartition(v_information_ratios_items, -k)[-k:])
#     else:
#         return np.sort(np.argpartition(v_information_ratios_items, -k)[-k:])