import numba
import numpy as np
from env import act_optimally
from collections import defaultdict

DISCRETIZATION_IDS = 25
RHO_VALUES = np.linspace(start=0.0, stop=1.0, num=DISCRETIZATION_IDS)


@numba.jit(nopython=True)
def g_full_numba(action, sampled_preferences, actions_star, counts, thetas):
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
        g_a += p_no_item_a_star_action * np.log(
            p_no_item_a_star_action / (p_star * p_no_item_action)
        )

    for ix in range(K):  # Now other y s are examined
        p_item_action = 0.0
        for m in range(M):
            p_item_action += probas_given_action[m, ix]
        p_item_action /= M
        if p_item_action > 1e-8:
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
                    g_a += p_item_a_star_action * np.log(
                        p_item_a_star_action / (p_star * p_item_action)
                    )
    if (probs < 0.999) or (probs > 1.001):
        raise ValueError("Problem in IDS with probabilities not summing to 1")
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
    result = 0.0
    for i in range(M):
        temp_sum = 0.0
        for k in range(K):
            temp_sum += pref[i, action[k]]
        result += temp_sum / (1 + temp_sum)
    return result / M


@numba.jit(nopython=True)
def delta_full_numba(action, sampled_preferences, r_star):
    """
    param: action 1D array of shape (K,) items selected in assortment
    param: preferences 2D array shape (M, N) sampled preferences
    param: r_star expected reward from taking optimal action
    for each theta model possible
    return: r_star - exp_reward
    """
    x = r_star - numba_expected_reward(action=action, pref=sampled_preferences)
    return x


@numba.jit(nopython=True)
def v_ids_numba(action, sampled_preferences, actions_star, counts, thetas):
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
        pref=sampled_preferences, action=action
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
def information_ratio_numba(rho, d1, d2, g1, g2):
    return (d1 * rho + (1 - rho) * d2) ** 2 / (g1 * rho + (1 - rho) * g2)


@numba.jit(nopython=True)
def information_difference(rho, d1, d2, v1, v2, eta):
    return (d1 * rho + (1 - rho) * d2) ** 2 - eta * (v1 * rho + (1 - rho) * v2)


@numba.jit(nopython=True)
def optimized_ratio_numba(d1, d2, g1, g2, scaler=0.0):
    n_rho = RHO_VALUES.shape[0]
    min_ = 1e8
    rho_min = -1
    for ix in range(n_rho):
        rho = RHO_VALUES[ix]
        if scaler:
            val = information_difference(rho, d1, d2, g1, g2, eta=scaler)
        else:
            val = information_ratio_numba(rho, d1, d2, g1, g2)
        if val < min_:
            rho_min = rho
            min_ = val
    return min_, rho_min


@numba.jit(nopython=True)
def ids_action_selection_numba(
    g_,
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
        delta_1 = delta_full_numba(action1, sampled_preferences, r_star)
        for j in range(n_actions):
            action2 = actions_set[j]
            g_a2 = g_(
                action2,
                sampled_preferences,
                actions_star,
                counts_star,
                thetas_star,
            )
            delta_2 = delta_full_numba(action2, sampled_preferences, r_star)
            g_a1 = 1e-12 if (not g_a1) else g_a1
            g_a2 = 1e-12 if (not g_a2) else g_a2
            value, rho = optimized_ratio_numba(
                d1=delta_1, d2=delta_2, g1=g_a1, g2=g_a2
            )
            action_picked = action1 if np.random.rand() <= rho else action2
            if value < min_information_ratio:
                min_information_ratio = value
                ids_action = action_picked
    return ids_action, min_information_ratio


@numba.jit(nopython=True)
def greedy_ids_action_selection(
    scaling_factor,
    g_,
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
    # Number of items and assortment size are key:
    n_items = sampled_preferences.shape[1]
    assortment_size = actions_star.shape[1]
    # Quantities to keep track off
    d1 = 0.0
    g1 = 0.0
    ids_action = np.zeros(assortment_size, dtype=np.int64)
    available_items = np.ones(n_items, dtype=np.int8)
    for current_size in range(assortment_size):
        current_action = np.copy(ids_action[: current_size + 1])
        min_information_difference = 1e8
        for item in range(n_items):
            if available_items[item]:
                current_action[current_size] = item
                current_delta = delta_full_numba(
                    current_action, sampled_preferences, r_star
                )
                current_g = g_(
                    current_action,
                    sampled_preferences,
                    actions_star,
                    counts_star,
                    thetas_star,
                )
                if not current_g:
                    value = current_delta
                else:
                    value = current_delta ** 2 - scaling_factor * current_g
                if value < min_information_difference:
                    min_information_difference = value
                    d1 = current_delta
                    g1 = current_g
                    ids_action[current_size] = item
        available_items[ids_action[current_size]] = 0
    if not g1:
        return ids_action
    else:
        rho_val = 0.5
        action2 = np.zeros(assortment_size, dtype=np.int64)
        available_items = np.ones(n_items, dtype=np.int8)
        for current_size in range(assortment_size):
            current_action = np.copy(action2[: current_size + 1])
            min_information_difference = 1e8
            for item in range(n_items):
                if available_items[item]:
                    current_action[current_size] = item
                    current_delta = delta_full_numba(
                        current_action, sampled_preferences, r_star
                    )
                    current_g = g_(
                        current_action,
                        sampled_preferences,
                        actions_star,
                        counts_star,
                        thetas_star,
                    )
                    value, rho = optimized_ratio_numba(
                        d1=d1,
                        d2=current_delta,
                        g1=g1,
                        g2=current_g,
                        scaler=scaling_factor,
                    )
                    if value < min_information_difference:
                        min_information_difference = value
                        action2[current_size] = item
                        rho_val = rho
            available_items[action2[current_size]] = 0
        action_picked = ids_action if np.random.rand() <= rho_val else action2
    return action_picked, 0.0


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
def greedy_mutual_information(
    items_forbidden,
    assortment_size,
    info_measure,
    prefs,
    a_star,
    count_star,
    thet_star,
):
    n_items = prefs.shape[1]
    n_excluded_0 = items_forbidden.shape[0]
    action_picked = np.zeros(assortment_size, dtype=np.int64)
    total_forbidden = -np.ones(n_excluded_0 + assortment_size, dtype=np.int64)
    total_forbidden[:n_excluded_0] = items_forbidden
    for current_size in range(1, assortment_size + 1):
        best_improvement = -np.inf
        item_added = -1
        for possible_item in range(n_items):
            action_picked[current_size - 1] = possible_item
            if insert_numba(
                total_forbidden[: (n_excluded_0 + current_size - 1)],
                possible_item,
            ):
                gain = info_measure(
                    action_picked[:current_size],
                    prefs,
                    a_star,
                    count_star,
                    thet_star,
                )
                if gain > best_improvement:
                    item_added = possible_item
                    best_improvement = gain
        action_picked[current_size - 1] = item_added
        total_forbidden[current_size - 1 + n_excluded_0] = item_added
    return action_picked


@numba.jit(nopython=True)
def ids_action_selection_approximate(
    g_,
    n_slots,
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
    # Pseudo code
    # 1- sort items by decreasing expected preference
    # 2- for k in range 1..K
    # pick items [1..k] in terms of pref
    # run ig_greedy to get items [k+1..K] in the assortment
    # 3- for k in rang 1..K
    # pick items [1..k] with ig_greedy
    # remaining items in terms of pref
    # optimize combination from those 2K actions
    # Also add logs to IDS in general
    expected_preferences = (
        np.sum(sampled_preferences, 0) / sampled_preferences.shape[0]
    )
    item_indexes_decreasing_expected_preferences = np.argsort(
        -expected_preferences
    )

    all_actions = np.ones((2 * n_slots, n_slots), dtype=np.int64)
    for k in range(1, n_slots + 1):
        items = item_indexes_decreasing_expected_preferences[:k]
        all_actions[k - 1, :k] = items
        other_items = greedy_mutual_information(
            items,
            n_slots - k,
            g_,
            sampled_preferences,
            actions_star,
            counts_star,
            thetas_star,
        )
        all_actions[k - 1, k:] = other_items

    for k in range(1, n_slots + 1):
        items = greedy_mutual_information(
            np.empty(shape=0, dtype=np.int64),
            k,
            g_,
            sampled_preferences,
            actions_star,
            counts_star,
            thetas_star,
        )
        all_actions[n_slots - 1 + k, :k] = items
        other_items = numba_top(
            item_indexes_decreasing_expected_preferences, items, n_slots - k
        )
        all_actions[n_slots - 1 + k, k:] = other_items

    return ids_action_selection_numba(
        g_, all_actions, sampled_preferences, r_star, actions_star, counts_star, thetas_star
    )


class InformationDirectedSampler:
    def __init__(self, assortment_size, n_samples, info_type):
        self.assortment_size = assortment_size
        self.n_samples = n_samples
        self.info_type = info_type
        self.init_sampler()

    def init_sampler(self):
        self.posterior_belief = np.random.rand(
            self.n_samples, self.assortment_size
        )
        self.optimal_actions = None
        self.actions_star = None
        self.thetas_star = None
        self.counts_star = None
        print(f"Info type is: {self.info_type}")
        if self.info_type == "gain":
            self.g_ = g_full_numba
        elif self.info_type == "variance":
            self.g_ = v_ids_numba
        else:
            raise ValueError("Choice of (gain | variance)")
        self.r_star = 0.0

    def update_r_star(self):
        sorted_beliefs = np.sort(self.posterior_belief, axis=1)[
            :, -self.assortment_size:
        ]  # shape (m, k)
        picking_probabilities = sorted_beliefs.sum(1)
        self.r_star = (
            picking_probabilities / (1 + picking_probabilities)
        ).mean()

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

    def update_belief(self, new_belief):
        self.posterior_belief = new_belief
        self.update_r_star()
        self.update_optimal_actions()
