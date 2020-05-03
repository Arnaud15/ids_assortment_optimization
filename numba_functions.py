import numba as nb
import numpy as np


@nb.jit(nopython=True)
def g_full_numba(action, sampled_preferences, opt_actions, counts, thetas):
    """
    :param action: array of actions indices of size K
    :param sampled_preferences: sampled posterior thetas
    :param opt_actions: vector of optimal actions
    :param probas
    :return:
    """
    g_a = 0.0
    probs = 0.0
    M = sampled_preferences.shape[0]
    K = action.shape[0]
    n_opt_actions = opt_actions.shape[0]

    probas_given_action = np.zeros((M, K))
    no_pick_given_action = np.zeros(shape=(M,))
    p_no_item_action = 0.0
    for i in range(M):
        sum_row = 0.0
        for ix in range(K):
            val = sampled_preferences[i, action[ix]]
            probas_given_action[i, ix] = val
            sum_row += val
        probas_given_action[i, :] = probas_given_action[i, :] / (1 + sum_row)
        no_pick_given_action[i] = 1 - (sum_row / (1 + sum_row))
        p_no_item_action += no_pick_given_action[i]
    p_no_item_action = p_no_item_action / M

    probs += p_no_item_action
    theta_start = 0
    for i in range(n_opt_actions):
        theta_indices = thetas[theta_start : theta_start + counts[i]]
        theta_start += counts[i]
        p_star = counts[i] / M
        p_no_item_a_star_action = 0.0
        for theta_indice in theta_indices:
            p_no_item_a_star_action += no_pick_given_action[theta_indice]
        p_no_item_a_star_action = p_no_item_a_star_action / M
        g_a += p_no_item_a_star_action * np.log(
            p_no_item_a_star_action / (p_star * p_no_item_action)
        )

    for ix in range(K):
        p_item_action = 0.0
        for m in range(M):
            p_item_action += probas_given_action[m, action[ix]]
        p_item_action /= M
        if p_item_action > 1e-5:
            probs += p_item_action
            theta_start = 0
            for j in range(n_opt_actions):
                p_star = counts[j] / M
                p_item_a_star_action = 0.0
                theta_indices = thetas[theta_start: theta_start + counts[j]]
                for theta_indice in theta_indices:
                    p_item_a_star_action += probas_given_action[
                        theta_indice, action[ix]
                    ]
                theta_start += counts[j]
                p_item_a_star_action /= M
                if p_item_a_star_action:
                    g_a += p_item_a_star_action * np.log(
                        p_item_a_star_action / (p_star * p_item_action)
                    )
    return g_a


@nb.jit(nopython=True)
def numba_expected_reward(pref, action):
    M = pref.shape[0]
    K = len(action)
    result = 0.0
    for i in range(M):
        temp_sum = 0.0
        for k in range(K):
            temp_sum += pref[i, action[k]]
        result += temp_sum / (1 + temp_sum)
    return result / M


@nb.jit(nopython=True)
def delta_full(action, sampled_preferences, r_star):
    x = r_star - numba_expected_reward(action=action, pref=sampled_preferences)
    return x
