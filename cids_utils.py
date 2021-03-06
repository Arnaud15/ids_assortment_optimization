import numpy as np
import cvxpy as cp
import logging
from env import act_optimally


def logar(arr):
    return (arr * 1000).astype(int) / 1000


def solve_cvx(regrets, variances, subset_size, n_items):
    x = cp.Variable(n_items, pos=True)
    deltas = cp.Parameter(n_items,)
    gains = cp.Parameter(n_items, pos=True)
    exp_delta = x @ deltas
    exp_gain = x @ gains
    information_ratio = cp.quad_over_lin(exp_delta, exp_gain)
    objective = cp.Minimize(information_ratio)
    constraints = [0 <= x, x <= 1, cp.sum(x) >= subset_size]
    prob = cp.Problem(objective, constraints,)
    deltas.value = regrets
    gains.value = variances

    try:
        prob.solve(solver="ECOS")
        zeros_index = x.value < 1e-3
        ones_index = x.value > 1 - 1e-3
        nzeros = zeros_index.sum()
        nones = ones_index.sum()
        nitems = x.value.shape[0]
        logging.debug(
            f"{nitems - nones - nzeros} nstrict, {nones} ones, {nzeros} zeroes, {nitems} total items"
        )
        if (nitems - nones - nzeros) == 2:
            all_items = np.arange(nitems)
            strict_items = all_items[~np.bitwise_or(zeros_index, ones_index)]
            probas = x.value[~np.bitwise_or(zeros_index, ones_index)]
            assert strict_items.shape[0] == 2, strict_items
            assert probas.shape[0] == 2, probas
            # 2 items to randomize the selection over
            logging.debug(f"items: {strict_items}, with probas: {probas}",)
            rho = probas[0]
            u = np.random.rand()
            if rho <= u:
                remaning_item = strict_items[0]
            else:
                remaning_item = strict_items[1]
            action = np.sort(
                np.concatenate(
                    [
                        act_optimally(x.value, top_k=subset_size - 1),
                        np.array([remaning_item]),
                    ]
                )
            )
        else:
            action = act_optimally(x.value, top_k=subset_size)
    except cp.SolverError:
        logging.warning("solver error")
        action = act_optimally(np.squeeze(-regrets), top_k=subset_size)
    except TypeError:
        logging.warning("solver error")
        action = act_optimally(np.squeeze(-regrets), top_k=subset_size)
    return action


def expected_regrets(posterior_belief, assortment_size):
    sorted_beliefs = np.sort(posterior_belief, axis=1)
    best_actions = sorted_beliefs[:, -assortment_size:]
    mean_rewards_best = best_actions.mean(1)  # mean over subset
    r_star = mean_rewards_best.mean()  # mean over samples
    expected_rewards = posterior_belief.mean(0)
    regrets = r_star - expected_rewards
    assert np.sort(regrets)[:assortment_size].sum() > 0.0
    return regrets


def var_if_a_star(posterior_belief, assortment_size):
    sorted_beliefs = np.sort(posterior_belief, axis=1)
    thresholds = sorted_beliefs[:, -assortment_size].reshape(-1, 1)
    expected_rewards = posterior_belief.mean(0)
    mask = posterior_belief >= thresholds
    p_star = mask.sum(0) / mask.shape[0]
    if_star = (posterior_belief * mask).sum(0) / (mask.sum(0) + 1e-12)
    variances = p_star * (if_star - expected_rewards) ** 2
    return np.maximum(variances, 1e-12)


def kl_if_a_star(posterior_belief, subset_size):
    """
    posterior_belief: [n_posterior_samples, n_items]
    1 <= subset_size <= n_items
    
    P(i \in A_star) D_kl(Y_i | i \in A_star || Y_i)

    with D_kl = p_0_star * log(p_0_star / p_0) + p_1 * log(p_1_star / p_1)

    p_0_star = 
    """
    sorted_beliefs = np.sort(posterior_belief, axis=1)
    thresholds = sorted_beliefs[:, -subset_size].reshape(-1, 1)
    mask = posterior_belief >= thresholds
    p_star = mask.sum(0) / mask.shape[0]
    if not p_star:
        return 1e-12
    p_1_star = (posterior_belief * mask).sum(0) / (mask.sum(0) + 1e-12)
    assert p_1_star > 0.0
    assert p_1_star < 1.0
    p_1 = posterior_belief.mean(0)
    kl_1 = p_1_star * np.log(p_1_star / p_1)
    kl_0 = (1.0 - p_1_star) * np.log((1.0 - p_1_star) / (1.0 - p_1))
    return kl_0 + kl_1
