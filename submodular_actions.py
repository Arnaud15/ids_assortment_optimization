from bintrees import FastRBTree
from scipy.special import xlogy
import numpy as np
import ipdb


def action_statistics(observations, n_items):
    action_to_id = {}
    count = 0
    probas = []
    zero_given_x = []
    for action, observation in observations:
        action = tuple(sorted(list(action)))
        if action in action_to_id:
            ix = action_to_id[action]
            probas[ix] += 1.0
            if observation == n_items:
                zero_given_x[ix] += 1.0
        else:
            action_to_id[action] = count
            probas.append(1.0)
            zero_given_x.append(1.0 if observation == n_items else 0)
            count += 1
    probas = np.array(probas)
    zero_given_x = np.array(zero_given_x)
    zero_given_x = zero_given_x / probas
    probas /= len(observations)
    assert probas.shape[0] == count
    assert zero_given_x.shape[0] == count
    assert probas.sum() < (1 + 1e-5)
    assert probas.sum() > (1 - 1e-5)
    id_to_action = {ix: action for (action, ix) in action_to_id.items()}
    return id_to_action, probas, zero_given_x


def f_function(t, p0):
    t = np.clip(t, 0.0, 1.0)
    return xlogy(t, t / p0) + xlogy(1 - t, (1 - t) / (1 - p0))


def query_improvement(idx, data, p_i, p_i_c, p0):
    s_left = data.prev_item(idx)[0]
    s_right = data.succ_item(idx)[0]
    p = p_i[idx] - p_i[s_left]
    q = p_i[s_right] - p_i[idx]
    alpha = (p_i[idx] * p_i_c[idx] - p_i[s_left] * p_i_c[s_left]) / p
    beta = (p_i[s_right] * p_i_c[s_right] - p_i[idx] * p_i_c[idx]) / q
    improv = (
        f_function(t=alpha, p0=p0) * p
        + f_function(t=beta, p0=p0) * q
        - (p + q) * f_function(t=(alpha * p + beta * q) / (p + q), p0=p0)
    )
    if np.isnan(improv):
        ipdb.set_trace()
    return (
        f_function(t=alpha, p0=p0) * p
        + f_function(t=beta, p0=p0) * q
        - (p + q) * f_function(t=(alpha * p + beta * q) / (p + q), p0=p0)
    )


def observations_to_actions(obs_run, n_items, m_actions):
    id_to_action, x_probas, zero_given_x = action_statistics(
        observations=obs_run, n_items=n_items
    )

    p_0 = len([x for (x, it) in obs_run if it == n_items]) / len(obs_run)
    sorting_indexes = np.argsort(zero_given_x)
    id_to_actions = [id_to_action[ix] for ix in sorting_indexes]
    zero_given_x = zero_given_x[sorting_indexes]
    x_probas = x_probas[sorting_indexes]

    n_actions = zero_given_x.shape[0]
    n_per_set = 2 * (n_actions // m_actions)
    S = FastRBTree()
    S.insert(0, 0)
    S.insert(n_actions, n_actions)
    p_inf = np.zeros(n_actions + 1)
    p_inf_c = np.zeros(n_actions + 1)
    for ix in range(1, n_actions + 1):
        p_inf[ix] = p_inf[ix - 1] + x_probas[ix - 1]
        p_inf_c[ix] = (
            p_inf[ix - 1] * p_inf_c[ix - 1]
            + zero_given_x[ix - 1] * x_probas[ix - 1]
        ) / p_inf[ix]

    indexes_available = np.ones(n_actions + 1, dtype=bool)
    indexes_available[n_actions] = False
    indexes_available[0] = False
    indexes = np.arange(n_actions + 1, dtype=int)
    for step_idx in range(m_actions - 1):
        largest_imp = -1
        index_added = None
        random_indexes = np.random.choice(
            indexes[indexes_available], size=n_per_set, replace=False
        )
        for index_candidate in random_indexes:
            S.insert(index_candidate, index_candidate)
            improvement = query_improvement(
                idx=index_candidate, data=S, p_i=p_inf, p_i_c=p_inf_c, p0=p_0
            )
            if improvement > largest_imp:
                largest_imp = improvement
                index_added = index_candidate
            S.remove(index_candidate)
        print(
            f"improvement {largest_imp}, with {index_added} at step {step_idx + 1}"
        )
        indexes_available[index_added] = False
        S.insert(index_added, index_added)

    return np.vstack(
        [np.array(id_to_actions[key - 1], dtype=int) for key in sorted(S.keys())[1:]]
    )
