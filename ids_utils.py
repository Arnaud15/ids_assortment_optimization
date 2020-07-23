import numba
from scipy.special import binom
import numpy as np
from env import act_optimally
from collections import defaultdict
import math

DISCRETIZATION_IDS = 11
RHO_VALUES = np.linspace(start=0.0, stop=1.0, num=DISCRETIZATION_IDS)


@numba.jit(nopython=True)
def g_theta(action, sampled_preferences, actions_star, counts, thetas):
    gain = 0.0
    probs = 0.0
    M = sampled_preferences.shape[0]
    K = action.shape[0]

    S_action = np.ones((M,), dtype=np.float64)
    for theta_ix in range(M):
        for item_ix in action:
            S_action[theta_ix] += sampled_preferences[theta_ix, item_ix]

    for item_ix in action:
        p_item = (sampled_preferences[:, item_ix] / S_action).mean()
        for theta_ix in range(M):
            p_item_given_theta = (
                sampled_preferences[theta_ix, item_ix] / S_action[theta_ix]
            )
            prob_step = p_item_given_theta / M
            probs += prob_step
            gain += prob_step * math.log(p_item_given_theta / p_item)

    p_no_item = (1 / (S_action)).mean()
    for theta_ix in range(M):
        p_no_item_given_theta = 1 / S_action[theta_ix]
        prob_step = p_no_item_given_theta / M
        probs += prob_step
        gain += prob_step * math.log(p_no_item_given_theta / p_no_item)

    if (probs < 1 - 1e-12) or (probs > 1 + 1e-12):
        raise ValueError("Problem in IDS with probabilities not summing to 1")
    return gain


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
    min_ = 1e12
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
    top_rho = 0.0
    top_g1 = 0.0
    top_r1 = 0.0
    top_g2 = 0.0
    top_r2 = 0.0
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
        delta_1 = delta_full_numba(action1, sampled_preferences, r_star,)
        for j in range(n_actions):
            action2 = actions_set[j]
            g_a2 = g_(
                action2,
                sampled_preferences,
                actions_star,
                counts_star,
                thetas_star,
            )
            delta_2 = delta_full_numba(action2, sampled_preferences, r_star,)
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
                top_g1 = g_a1
                top_r1 = delta_1
                top_g2 = g_a2
                top_r2 = delta_2
    return (
        ids_action,
        min_information_ratio,
        top_rho,
        top_g1,
        top_r1,
        top_g2,
        top_r2,
    )


@numba.jit(nopython=True)
def greedy_information_difference(
    starting_action,
    available_items,
    action_size,
    g_,
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
                current_delta = delta_full_numba(
                    current_action, thetas, r_star
                )
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
def greedy_regret_minimization(
    starting_action,
    available_items,
    action_size,
    g_,
    thetas,
    actions_star,
    counts_star,
    thetas_star,
    r_star,
    entropy,
    lambda_cst,
    time_left,
    n_items,
    d1,
    g1,
):
    delta_val = 0.0
    g_val = 0.0
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
                current_delta = delta_full_numba(
                    current_action, thetas, r_star
                )
                current_g = g_(
                    current_action,
                    thetas,
                    actions_star,
                    counts_star,
                    thetas_star,
                )
                current_g = current_g if current_g > 1e-12 else 1e-12
                value = current_delta + np.sqrt(
                    lambda_cst * (entropy - current_g) * time_left
                )
                value += np.random.rand() * 1e-12  # adding random noise
                if value < min_information:
                    min_information = value
                    delta_val = current_delta
                    g_val = current_g
                    starting_action[current_size - 1] = item
        available_items[starting_action[current_size - 1]] = 0
    return starting_action, delta_val, g_val


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
    n_items = sampled_preferences.shape[1]
    assortment_size = actions_star.shape[1]
    ids_action = -np.ones(assortment_size, dtype=np.int64)
    available_items = np.ones(n_items, dtype=np.int8)
    action_1, d_1, g_1, _ = greedy_information_difference(
        starting_action=ids_action,
        available_items=available_items,
        action_size=assortment_size,
        g_=g_,
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
        information_ratio_numba(rho=rho_val, d1=d_1, d2=d_2, g1=g_1, g2=g_2),
        rho_val,
        g_1,
        d_1,
        g_2,
        d_2,
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
def to_key(arr):
    total = 0
    ix = 0
    for elt in np.sort(arr):
        total += elt * 10 ** ix
        ix += 1
    return total


@numba.jit(nopython=True)
def ids_action_selection_approximate(
    entropy,
    lambda_cst,
    time_left,
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
    n_items = sampled_preferences.shape[1]
    assortment_size = actions_star.shape[1]
    ids_action = -np.ones(assortment_size, dtype=np.int64)
    available_items = np.ones(n_items, dtype=np.int8)
    action_1, d_1, g_1 = greedy_regret_minimization(
        starting_action=ids_action,
        available_items=available_items,
        action_size=assortment_size,
        g_=g_,
        thetas=sampled_preferences,
        actions_star=actions_star,
        counts_star=counts_star,
        thetas_star=thetas_star,
        entropy=entropy,
        time_left=time_left,
        lambda_cst=lambda_cst,
        r_star=r_star,
        n_items=n_items,
        d1=0.0,
        g1=0.0,
    )
    return (
        action_1,
        d_1 ** 2 / g_1 if g_1 > 1e-10 else 0.0,
        0.5,
        g_1,
        d_1,
        g_1,
        d_1,
    )


class InformationDirectedSampler:
    def __init__(self, n_items, assortment_size, n_samples, info_type):
        self.n_items = n_items
        self.assortment_size = assortment_size
        self.n_samples = n_samples
        self.max_entropy = assortment_size * np.log(n_items / assortment_size)
        self.max_s_entropy = np.log(n_samples)
        self.info_type = info_type
        self.n_possible_actions = binom(n_items, assortment_size)
        self.init_sampler()

    def init_sampler(self):
        self.posterior_belief = np.random.rand(self.n_samples, self.n_items)
        self.max_ir = 0.2
        self.optimal_actions = None
        self.actions_star = None
        self.a_star_entropy = 0.0
        self.thetas_star = None
        self.counts_star = None
        print(f"Info type is: {self.info_type}")
        if self.info_type == "gain":
            self.g_ = g_full_numba
        elif self.info_type == "variance":
            self.g_ = v_ids_numba
        elif self.info_type == "gain_theta":
            self.g_ = g_theta
        else:
            raise ValueError("Choice of (gain | variance | gain_theta)")
        self.r_star = 0.0
        self.update_belief(self.posterior_belief)
        self.lambda_min = 1.0

    def update_r_star(self):
        sorted_beliefs = np.sort(self.posterior_belief, axis=1)[
            :, -self.assortment_size :
        ]  # shape (m, k)
        picking_probabilities = sorted_beliefs.sum(1)
        self.r_star = (
            picking_probabilities / (1 + picking_probabilities)
        ).mean()
        a_greedy = act_optimally(
            self.posterior_belief.mean(0), self.assortment_size
        )
        greedy_expected_reward = self.posterior_belief[:, a_greedy]
        greedy_expected_reward = greedy_expected_reward.sum(1)
        greedy_expected_reward = greedy_expected_reward / (
            1.0 + greedy_expected_reward
        )
        greedy_expected_reward = greedy_expected_reward.mean()
        self.delta_min = (self.r_star - greedy_expected_reward) ** 2
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

    def update_belief(self, new_belief):
        self.posterior_belief = new_belief
        self.update_r_star()
        self.update_optimal_actions()
        if self.a_star_entropy < 1e-12:
            self.lambda_algo = 0.0
        else:
            self.lambda_algo = self.delta_min / self.a_star_entropy

    def update_lambda(self):
        min_val = 1e12
        for _ in range(10):
            a1 = np.random.choice(
                a=self.n_items, size=self.assortment_size, replace=False
            )
            a2 = np.random.choice(
                a=self.n_items, size=self.assortment_size, replace=False
            )
            g_1 = self.g_(
                a1,
                sampled_preferences=self.posterior_belief,
                actions_star=self.actions_star,
                counts=self.counts_star,
                thetas=self.thetas_star,
            )
            g_1 = max(g_1, 1e-12)
            g_2 = self.g_(
                a2,
                sampled_preferences=self.posterior_belief,
                actions_star=self.actions_star,
                counts=self.counts_star,
                thetas=self.thetas_star,
            )
            g_2 = max(g_2, 1e-12)
            d_1 = delta_full_numba(a1, self.posterior_belief, self.r_star)
            d_2 = delta_full_numba(a2, self.posterior_belief, self.r_star)
            val = optimized_ratio_numba(d_1, d_2, g_1, g_2)[0]
            if val < min_val:
                min_val = val
        self.lambda_min = min_val
