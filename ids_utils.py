import numba
from scipy.special import binom
from scipy.stats import beta
import numpy as np
from env import act_optimally
from collections import defaultdict
import math

DISCRETIZATION_IDS = 11
RHO_VALUES = np.linspace(start=0.0, stop=1.0, num=DISCRETIZATION_IDS)


def bernoulli_entropy(proba):
    clipped_proba = np.clip(proba, a_min=1e-12, a_max=1 - 1e-12)
    return -(
        np.log(clipped_proba) * proba + np.log(1 - clipped_proba) * (1 - proba)
    )


def compute_info_gains(
    posterior_parameters, assortment_size, n_items, n_samples
):
    a_s = posterior_parameters[0].reshape(1, -1)
    b_s = posterior_parameters[1].reshape(1, -1)
    base_samples = np.random.rand(n_samples, n_items)
    samples_before = (1 / beta.isf(a=a_s, b=b_s, q=base_samples)) - 1
    samples_after = (1 / beta.isf(a=a_s + 1, b=b_s, q=base_samples)) - 1
    sorted_beliefs = np.sort(samples_before, axis=1)
    thresholds = sorted_beliefs[:, -assortment_size].reshape(-1, 1)
    mask_before = samples_before >= thresholds
    p_before = mask_before.sum(0) / mask_before.shape[0]
    mask_after = samples_after >= thresholds
    p_after = mask_after.sum(0) / mask_after.shape[0]
    entropy_before = bernoulli_entropy(p_before)
    entropy_after = bernoulli_entropy(p_after)
    return np.clip(entropy_before - entropy_after, a_min=1e-12, a_max=None)


def approximate_info_gain(vi_s, ni_s):
    proba = vi_s / ni_s
    return proba * (proba + 1.0) / (ni_s * (ni_s + 1.0))


def simplex_action_selection(n, k, x):
    x += np.random.rand(x.shape[0]) * 1e-8
    v1 = max(0, k - 1)
    v2 = min([n - v1, 2, k])
    sortem_item_ixs = np.argsort(x)
    deterministic = sortem_item_ixs[-v1:][:v1]
    action = None
    if not v2:
        action = np.sort(deterministic)
    elif v2 == 1:
        last_action = sortem_item_ixs[-v1 - 1]
        action = np.sort(np.append(deterministic, last_action))
    else:
        i1, i2 = sortem_item_ixs[-v1 - v2 : (-v1)]
        p1, p2 = x[[i1, i2]]
        np1 = p1 / (p1 + p2 + 1e-12)
        rho = np.random.rand()
        last_action = i1 if rho <= np1 else i2
        action = np.sort(np.append(deterministic, last_action))
    return action


@numba.jit(nopython=True)
def info_gain_step(action, sampled_preferences, actions_star, counts, thetas):
    """
    :param action: 1D array of size (K,) with the indices of the current action
    :param sampled_preferences: sampled posterior thetas of shape (M, N)
    :param actions_star: 2D array of shape (n_top_actions, assortment_size)
    :param counts: 1D array of shape (n_top_actions,)
     = how many theta for each opt action
    :param thetas: 1D array of indices in [0, M-1]
    of the thetas associated w/ each opt action
    :return:
    """
    g_a = 0.0
    probs = 0.0
    M = sampled_preferences.shape[0]
    K = action.shape[0]
    n_actions_star = actions_star.shape[0]

    probas_given_action = np.zeros(
        (M, K)
    )  # Probabilities of each item given action
    no_pick_given_action = np.zeros((M,))  # Same for no pick given action
    p_no_item_action = 0.0
    for m in range(M):
        sum_row = 0.0
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
    for i in range(
        n_actions_star
    ):  # First we treat separately the y=NO_ITEM case
        theta_indices = thetas[theta_start : theta_start + counts[i]]
        theta_start += counts[i]
        p_star = counts[i] / M
        p_no_item_a_star_action = 0.0
        for theta_indice in theta_indices:
            p_no_item_a_star_action += no_pick_given_action[theta_indice]
        p_no_item_a_star_action = p_no_item_a_star_action / M
        g_a += p_no_item_a_star_action * math.log(
            p_no_item_a_star_action / (p_star * p_no_item_action)
        )

    for ix in range(K):  # Now other y s are examined
        p_item_action = probas_given_action[:, ix].mean()
        if p_item_action > 1e-12:
            probs += p_item_action
            theta_start = 0
            for j in range(n_actions_star):
                p_star = counts[j] / M
                p_item_a_star_action = 0.0
                theta_indices = thetas[theta_start : theta_start + counts[j]]
                for theta_indice in theta_indices:
                    p_item_a_star_action += probas_given_action[
                        theta_indice, ix
                    ]
                theta_start += counts[j]
                p_item_a_star_action /= M
                if p_item_a_star_action:
                    g_a += p_item_a_star_action * math.log(
                        p_item_a_star_action / (p_star * p_item_action)
                    )
    if (probs < 1 - 1e-12) or (probs > 1 + 1e-12):
        raise ValueError("Problem in IDS with probabilities not summing to 1")
    return g_a


@numba.jit(nopython=True)
def numba_expected_reward(pref, action, mode):
    """
    param: pref 2D array shape (M, N) sampled preferences
    param: action 1D array of shape (K,) items selected in assortment
    return: expected reward for action = scalar
    """
    M = pref.shape[0]
    K = action.shape[0]
    result = 0.0
    for i in range(M):
        temp_sum = 0.0
        for k in range(K):
            temp_sum += pref[i, action[k]]
        if mode == "step":
            result += temp_sum / (1 + temp_sum)
        else:
            assert mode == "epoch"
            result += temp_sum
    return result / M


@numba.jit(nopython=True)
def delta_step(action, sampled_preferences, r_star):
    """
    param: action 1D array of shape (K,) items selected in assortment
    param: preferences 2D array shape (M, N) sampled preferences
    param: r_star expected reward from taking optimal action
    for each theta model possible
    return: r_star - exp_reward
    """
    x = r_star - numba_expected_reward(
        action=action, pref=sampled_preferences, mode="step"
    )
    return x


@numba.jit(nopython=True)
def vids_step(action, sampled_preferences, actions_star, counts, thetas):
    """
    :param action: 1D array of size (K,) with the indices of the current action
    :param sampled_preferences: sampled posterior thetas of shape (M, N)
    :param actions_star: 2D array of shape (n_top_actions, assortment_size)
    :param counts: 1D array of shape (n_top_actions,)
    = how many theta for each opt action
    :param thetas: 1D array of the indices in [0, M-1]
    of the thetas associated w/ each opt action
    :return:
    """
    expected_reward_action = numba_expected_reward(
        pref=sampled_preferences, action=action, mode="step"
    )
    M = sampled_preferences.shape[0]
    K = action.shape[0]
    n_opt_actions = actions_star.shape[0]

    pick_given_action = np.zeros(shape=(M,))
    for i in range(M):
        sum_row = 0.0
        for ix in range(K):
            sum_row += sampled_preferences[i, action[ix]]
        pick_given_action[i] = sum_row / (1 + sum_row)

    probs = 0.0
    theta_start = 0
    v_a = 0
    for j in range(n_opt_actions):
        p_star = counts[j] / M
        p_pick_a_star_action = 0.0
        theta_indices = thetas[theta_start : theta_start + counts[j]]
        for theta_indice in theta_indices:
            p_pick_a_star_action += pick_given_action[theta_indice]
        theta_start += counts[j]
        p_pick_a_star_action /= counts[j]
        v_a += p_star * (p_pick_a_star_action - expected_reward_action) ** 2
        probs += p_star
    if (probs < 0.999) or (probs > 1.001):
        raise ValueError("Problem in VIDS with probabilities not summing to 1")
    return v_a


@numba.jit(nopython=True)
def vids_epoch(action, sampled_preferences, actions_star, counts, thetas):
    M = sampled_preferences.shape[0]
    K = action.shape[0]
    n_opt_actions = actions_star.shape[0]

    expected_reward_action = numba_expected_reward(
        pref=sampled_preferences, action=action, mode="epoch"
    )

    reward_given_thetas = np.zeros(shape=(M,))
    for i in range(M):
        sum_row = 0.0
        for ix in range(K):
            sum_row += sampled_preferences[i, action[ix]]
        reward_given_thetas[i] = sum_row

    probs = 0.0
    theta_start = 0
    v_a = 0.0
    for j in range(n_opt_actions):
        p_star = counts[j] / M
        reward_given_a_star = 0.0
        theta_indices = thetas[theta_start : theta_start + counts[j]]
        for theta_indice in theta_indices:
            reward_given_a_star += reward_given_thetas[theta_indice]
        theta_start += counts[j]
        reward_given_a_star /= counts[j]
        v_a += p_star * (reward_given_a_star - expected_reward_action) ** 2
        probs += p_star
    if (probs < 0.999) or (probs > 1.001):
        raise ValueError("Problem in VIDS with probabilities not summing to 1")
    return v_a


@numba.jit(nopython=True)
def info_gain_epoch(action, sampled_preferences, actions_star, counts, thetas):
    """
    :param action: 1D array of size (K,) with the indices of the current action
    :param sampled_preferences: sampled posterior thetas of shape (M, N)
    :param actions_star: 2D array of shape (n_top_actions, assortment_size)
    :param counts: 1D array of shape (n_top_actions,)
     = how many theta for each opt action
    :param thetas: 1D array of indices in [0, M-1]
    of the thetas associated w/ each opt action
    :return:
    """
    g_a = 0.0
    probs = 0.0
    M = sampled_preferences.shape[0]
    K = action.shape[0]
    n_actions_star = actions_star.shape[0]

    probas_given_action = np.zeros(
        (M, K)
    )  # Probabilities of each item given action
    no_pick_given_action = np.zeros((M,))  # Same for no pick given action
    p_no_item_action = 0.0
    for m in range(M):
        sum_row = 0.0
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
    for i in range(
        n_actions_star
    ):  # First we treat separately the y=NO_ITEM case
        theta_indices = thetas[theta_start : theta_start + counts[i]]
        theta_start += counts[i]
        p_star = counts[i] / M
        p_no_item_a_star_action = 0.0
        for theta_indice in theta_indices:
            p_no_item_a_star_action += no_pick_given_action[theta_indice]
        p_no_item_a_star_action = p_no_item_a_star_action / M
        g_a += p_no_item_a_star_action * math.log(
            p_no_item_a_star_action / (p_star * p_no_item_action)
        )

    for ix in range(K):  # Now other y s are examined
        p_item_action = probas_given_action[:, ix].mean()
        if p_item_action > 1e-12:
            probs += p_item_action
            theta_start = 0
            for j in range(n_actions_star):
                p_star = counts[j] / M
                p_item_a_star_action = 0.0
                theta_indices = thetas[theta_start : theta_start + counts[j]]
                for theta_indice in theta_indices:
                    p_item_a_star_action += probas_given_action[
                        theta_indice, ix
                    ]
                theta_start += counts[j]
                p_item_a_star_action /= M
                if p_item_a_star_action:
                    g_a += p_item_a_star_action * math.log(
                        p_item_a_star_action / (p_star * p_item_action)
                    )
    if (probs < 1 - 1e-12) or (probs > 1 + 1e-12):
        raise ValueError("Problem in IDS with probabilities not summing to 1")
    return g_a


@numba.jit(nopython=True)
def delta_epoch(action, sampled_preferences, r_star):
    x = r_star - numba_expected_reward(
        action=action, pref=sampled_preferences, mode="epoch"
    )
    return x


@numba.jit(nopython=True)
def rewards_table(sampled_preferences):
    M = sampled_preferences.shape[0]
    N = sampled_preferences.shape[1]
    result = np.zeros((N,), dtype=np.float64)
    for n in range(N):
        result[n] = sampled_preferences[:, n].mean()
    return result


@numba.jit(nopython=True)
def gains_table(L, sampled_preferences, actions_star, counts, thetas):
    N = sampled_preferences.shape[1]
    M = sampled_preferences.shape[0]
    A = actions_star.shape[0]
    g_table_base = np.zeros((N, M, L), dtype=np.float64)
    for item_ix in range(N):
        for theta_ix in range(M):
            geom_params_item = 1.0 / (
                sampled_preferences[theta_ix, item_ix] + 1.0
            )
            proba = geom_params_item
            sum_probs = 0.0
            for n_picks in range(L):
                g_table_base[item_ix, theta_ix, n_picks] = proba
                sum_probs += proba
                proba *= 1 - geom_params_item
    g_table = np.zeros((N,), dtype=np.float64)
    for item_ix in range(N):
        g_item = 0.0
        p_picks = np.zeros((L,), dtype=np.float64)
        for l in range(L):
            p_picks[l] = g_table_base[item_ix, :, l].mean()
        theta_start = 0
        for action_star_ix in range(A):
            p_star = counts[action_star_ix] / M
            p_a_star_picks = np.zeros((L,), dtype=np.float64)
            theta_indices = thetas[
                theta_start : theta_start + counts[action_star_ix]
            ]
            for theta_indice in theta_indices:
                p_a_star_picks += g_table_base[item_ix, theta_indice, :]
            theta_start += counts[action_star_ix]
            p_a_star_picks /= counts[action_star_ix]
            g_item += (
                p_star
                * p_a_star_picks
                * np.log((p_a_star_picks + 1e-12) / (p_picks + 1e-12))
            ).sum()
        g_table[item_ix] = g_item
    return g_table


@numba.jit(nopython=True)
def information_ratio(rho, d1, d2, g1, g2):
    return (d1 * rho + (1 - rho) * d2) ** 2 / (g1 * rho + (1 - rho) * g2)


@numba.jit(nopython=True)
def information_difference(rho, d1, d2, v1, v2, eta):
    return (d1 * rho + (1 - rho) * d2) ** 2 - eta * (v1 * rho + (1 - rho) * v2)


@numba.jit(nopython=True)
def optimized_ratio_numba(d1, d2, g1, g2, scaler=0.0):
    n_rho = RHO_VALUES.shape[0]
    min_ = 1e12
    rho_min = -1
    for ix in range(n_rho):
        rho = RHO_VALUES[ix]
        if scaler:
            val = information_difference(rho, d1, d2, g1, g2, eta=scaler)
        else:
            val = information_ratio(rho, d1, d2, g1, g2)
        if val < min_:
            rho_min = rho
            min_ = val
    return min_, rho_min


@numba.jit(nopython=True)
def ids_exact_action(
    g_,
    d_,
    actions_set,
    sampled_preferences,
    r_star,
    actions_star,
    counts_star,
    thetas_star,
):
    """
    param: g_ = information_ratio computation function selected
    param: actions_set = possible actions given n, k
    param: preferences 2D array shape (M, N) sampled preferences
    param: r_star expected reward from taking optimal action
    for each theta model possible
    :param actions_star: 2D array of shape (n_top_actions, assortment_size)
    :param counts: 1D array of shape (n_top_actions,)
    = how many theta for each opt action
    :param thetas: 1D array of the indices in [0, M-1]
    of the thetas associated w/ each opt action
    """
    # Shuffling the actions set
    np.random.shuffle(actions_set)
    # Quantities to keep track off
    min_information_ratio = 1e8
    ids_action = actions_set[0]
    top_rho = 0.0
    n_actions = actions_set.shape[0]
    for i in range(n_actions):
        action1 = actions_set[i]
        g_a1 = g_(
            action1,
            sampled_preferences,
            actions_star,
            counts_star,
            thetas_star,
        )
        delta_1 = d_(action1, sampled_preferences, r_star,)
        for j in range(n_actions):
            action2 = actions_set[j]
            g_a2 = g_(
                action2,
                sampled_preferences,
                actions_star,
                counts_star,
                thetas_star,
            )
            delta_2 = d_(action2, sampled_preferences, r_star,)
            g_a1 = 1e-12 if g_a1 < 1e-12 else g_a1
            g_a2 = 1e-12 if g_a2 < 1e-12 else g_a2
            value, rho = optimized_ratio_numba(
                d1=delta_1, d2=delta_2, g1=g_a1, g2=g_a2
            )
            action_picked = action1 if np.random.rand() <= rho else action2
            if value < min_information_ratio:
                min_information_ratio = value
                ids_action = action_picked
                top_rho = rho
    return (
        ids_action,
        min_information_ratio,
        top_rho,
    )


@numba.jit(nopython=True)
def greedy_information_difference(
    starting_action,
    available_items,
    action_size,
    g_,
    d_,
    thetas,
    actions_star,
    counts_star,
    thetas_star,
    r_star,
    scaler,
    n_items,
    mixing,
    d1,
    g1,
):
    delta_val = 0.0
    g_val = 0.0
    rho_mixing = -1.0
    for current_size in range(1, action_size + 1):
        min_information = 1e12
        if starting_action[current_size - 1] < 0:
            starting_action[current_size - 1] = 0
        else:
            available_items[starting_action[current_size - 1]] = 1
        current_action = np.copy(starting_action[starting_action >= 0])
        for item in range(n_items):
            if available_items[item]:
                current_action[current_size - 1] = item
                current_delta = d_(current_action, thetas, r_star)
                current_g = g_(
                    current_action,
                    thetas,
                    actions_star,
                    counts_star,
                    thetas_star,
                )
                current_g = current_g if current_g > 1e-12 else 1e-12
                if mixing:
                    value, current_rho = optimized_ratio_numba(
                        d1=d1,
                        d2=current_delta,
                        g1=g1,
                        g2=current_g,
                        scaler=scaler,
                    )
                else:
                    value = current_delta ** 2 - scaler * current_g
                    current_rho = -1.0
                value += np.random.rand() * 1e-12  # adding random noise
                if value < min_information:
                    min_information = value
                    delta_val = current_delta
                    g_val = current_g
                    starting_action[current_size - 1] = item
                    rho_mixing = current_rho
        available_items[starting_action[current_size - 1]] = 0
    return starting_action, delta_val, g_val, rho_mixing


@numba.jit(nopython=True)
def greedy_ids_action(
    scaling_factor,
    g_,
    d_,
    sampled_preferences,
    r_star,
    actions_star,
    counts_star,
    thetas_star,
):
    """
    param: g_ = information_ratio computation function selected
    param: d_ = regret function 
    param: actions_set = possible actions given n, k
    param: preferences 2D array shape (M, N) sampled preferences
    param: r_star expected reward from taking optimal action
    for each theta model possible
    :param actions_star: 2D array of shape (n_top_actions, assortment_size)
    :param counts: 1D array of shape (n_top_actions,)
    = how many theta for each opt action
    :param thetas: 1D array of the indices in [0, M-1]
    of the thetas associated w/ each opt action
    """
    n_items = sampled_preferences.shape[1]
    assortment_size = actions_star.shape[1]
    ids_action = -np.ones(assortment_size, dtype=np.int64)
    available_items = np.ones(n_items, dtype=np.int8)
    action_1, d_1, g_1, _ = greedy_information_difference(
        starting_action=ids_action,
        available_items=available_items,
        action_size=assortment_size,
        g_=g_,
        d_=d_,
        thetas=sampled_preferences,
        actions_star=actions_star,
        counts_star=counts_star,
        thetas_star=thetas_star,
        scaler=scaling_factor,
        r_star=r_star,
        n_items=n_items,
        mixing=False,
        d1=0.0,
        g1=0.0,
    )

    action_2, d_2, g_2, rho_val = greedy_information_difference(
        starting_action=-np.ones(assortment_size, dtype=np.int64),
        available_items=np.ones(n_items, dtype=np.int8),
        action_size=assortment_size,
        g_=g_,
        d_=d_,
        thetas=sampled_preferences,
        actions_star=actions_star,
        counts_star=counts_star,
        thetas_star=thetas_star,
        scaler=scaling_factor,
        r_star=r_star,
        n_items=n_items,
        mixing=True,
        d1=d_1,
        g1=g_1,
    )
    ids_action = action_1 if np.random.rand() <= rho_val else action_2
    return (
        ids_action,
        information_ratio(rho=rho_val, d1=d_1, d2=d_2, g1=g_1, g2=g_2),
        rho_val,
    )


@numba.jit(nopython=True)
def insert_numba(forbidden, current_elt):
    for elt in forbidden:
        if current_elt == elt:
            return False
    return True


@numba.jit(nopython=True)
def numba_top(input_arr, forbidden, size):
    count = 0
    output_top = np.zeros(size, dtype=np.int64)
    for ix in range(input_arr.shape[0]):
        current_elt = input_arr[ix]
        insert = insert_numba(forbidden, current_elt)
        if insert:
            output_top[count] = current_elt
            count += 1
        if count >= size:
            break
    return output_top


@numba.jit(nopython=True)
def to_key(arr):
    total = 0
    ix = 0
    for elt in np.sort(arr):
        total += elt * 10 ** ix
        ix += 1
    return total


class InformationDirectedSampler:
    def __init__(
        self, n_items, assortment_size, n_samples, info_type, dynamics
    ):
        self.n_items = n_items
        self.assortment_size = assortment_size
        self.n_samples = n_samples
        self.info_type = info_type
        self.dynamics = dynamics
        self.max_entropy = assortment_size * np.log(n_items / assortment_size)
        self.max_s_entropy = np.log(n_samples)
        self.n_possible_actions = binom(n_items, assortment_size)
        self.init_sampler()

    def init_sampler(self):
        self.posterior_belief = np.random.rand(self.n_samples, self.n_items)
        self.optimal_actions = None
        self.actions_star = None
        self.thetas_star = None
        self.counts_star = None
        if self.dynamics == "epoch":
            assert (
                self.info_type == "variance"
            ), "Gain not supported for epoch yet."
            self.g_ = vids_epoch
            self.d_ = delta_epoch
        elif self.dynamics == "step":
            self.d_ = delta_step
            if self.info_type == "gain":
                self.g_ = info_gain_step
            elif self.info_type == "variance":
                self.g_ = vids_step
        else:
            raise ValueError("Choice of (epoch - step)")
        self.r_star = 0.0
        self.update_belief(self.posterior_belief)
        self.a_star_entropy = self.max_entropy

    def update_r_star(self):
        sorted_beliefs = np.sort(self.posterior_belief, axis=1)[
            :, -self.assortment_size :
        ]  # shape (m, k)
        picking_probabilities = sorted_beliefs.sum(1)
        if self.dynamics == "epoch":
            self.r_star = picking_probabilities.mean()
        else:
            self.r_star = (
                picking_probabilities / (1 + picking_probabilities)
            ).mean()
        a_greedy = act_optimally(
            self.posterior_belief.mean(0), self.assortment_size
        )
        greedy_expected_reward = numba_expected_reward(
            self.posterior_belief, a_greedy, mode=self.dynamics
        )
        self.delta_min = self.r_star - greedy_expected_reward
        assert self.delta_min > -1e-12, (
            self.delta_min,
            self.r_star,
            greedy_expected_reward,
        )
        self.delta_min = max(1e-12, self.delta_min)
        # print(
        #     f"r_star = {self.r_star:.2f}, greedy action expected reward = {greedy_expected_reward:.2f}, delta min = {self.delta_min:.2f}"
        # )

    def update_optimal_actions(self):
        """
        :return: dictionary of informations about optimal action
        for each posterior sample of the model parameters
        # keys: actions = sorted tuple of items to propose in the assortment
        # values: (p(action = a*),
        [thetas such that action is optimal for theta]
        """
        posteriors_actions = act_optimally(
            self.posterior_belief, self.assortment_size
        )
        posteriors_actions = [
            tuple(posteriors_actions[ix, :]) for ix in range(self.n_samples)
        ]
        optimal_actions_information = defaultdict(list)
        for ix, action in enumerate(posteriors_actions):
            optimal_actions_information[action].append(ix)

        self.optimal_actions = {
            action: (len(theta_idxs) / self.n_samples, theta_idxs)
            for action, theta_idxs in optimal_actions_information.items()
        }
        self.actions_star = np.array(
            [list(key) for key in optimal_actions_information.keys()]
        )
        self.counts_star = np.array(
            [len(val) for val in optimal_actions_information.values()]
        )
        self.thetas_star = []
        for val in optimal_actions_information.values():
            self.thetas_star += val
        self.thetas_star = np.array(self.thetas_star)
        self.a_star_entropy = sum(
            [
                -p * np.log(p)
                for (action, (p, _)) in self.optimal_actions.items()
                if p > 0.0
            ]
        )
        self.a_star_entropy = (
            self.max_entropy * self.a_star_entropy / self.max_s_entropy
        )
        # print(
        #     f"Updated entropy of the optimal action distribution is {self.a_star_entropy:.2f} which is {self.a_star_entropy / np.log(self.n_possible_actions):.2f} percent of total randomness."
        # )

    def update_tables(self):
        # if self.dynamics == "epoch" and self.info_type == "gain"
        self.gs_table = gains_table(
            self.assortment_size,
            self.posterior_belief,
            self.actions_star,
            self.counts_star,
            self.thetas_star,
        )
        self.rews_table = rewards_table(self.posterior_belief)

    def update_belief(self, new_belief):
        self.posterior_belief = new_belief
        self.update_r_star()
        self.update_optimal_actions()
        self.update_tables()
        if self.a_star_entropy < 1e-12:
            self.lambda_algo = 0.0
        else:
            self.lambda_algo = self.delta_min ** 2 / self.a_star_entropy
