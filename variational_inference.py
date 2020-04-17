import numba
import numpy as np


@numba.jit(nopython=True)
def elbo(xi, mu, L, prior_std, picks, assortments):
    n = L.shape[0]
    k = assortments.shape[1]
    timesteps = picks.shape[0]
    sigma = L ** 2
    mu_gradient = - (mu - xi) / (prior_std ** 2)
    L_gradient = np.ones(n) / (prior_std ** 2)
    
    q_entropy = 0
    for i in range(n):
        q_entropy += np.log(np.abs(L[i]))
    
    cross_entropy = - 0.5 * np.sum( (sigma + (mu - xi) **2) / (prior_std ** 2))
    
    log_likelihood = 0
    for t in range(timesteps):
        item_picked = picks[t]

        if item_picked < n:
            log_likelihood += mu[item_picked]
            mu_gradient[item_picked] += 1

        assortment_weight = 1.
        for j in range(k):
            item_j = assortments[t, j]
            assortment_weight += np.exp(mu[item_j] + 0.5 * sigma[item_j])
    
        for j in range(k):
            item_j = assortments[t, j]
            w_tj = np.exp(mu[item_j] + 0.5 * sigma[item_j]) / assortment_weight
            mu_gradient[item_j] -= w_tj
            L_gradient[item_j] += w_tj
        
        log_likelihood -= np.log(assortment_weight)
    
    L_gradient = - L * L_gradient
    for i in range(n):
        L_gradient[i] += 1 / L[i]

#     print(cross_entropy, q_entropy, log_likelihood)
#     print(mu_gradient, L_gradient)
    return cross_entropy + q_entropy + log_likelihood, mu_gradient, L_gradient


@numba.jit(nopython=True)
def variational_update(mu, L, items_picked, assortments):
    PRIOR_STD = 0.5
    PRIOR_MEAN = - 2 / 3
    STEP_SIZE = 1e-4
    past_objective_value = 0
    xi = np.ones(mu.shape[0]) * PRIOR_MEAN
    for _ in range(250):
        objective, mu_g, l_g = elbo(xi=xi,
                mu=mu,
                L=L,
                prior_std=PRIOR_STD,
                picks=items_picked,
                assortments=assortments)
        if (np.abs(past_objective_value - objective)) < 1e-3:
            return mu, L
        else:
            mu = mu + STEP_SIZE * mu_g
            L = L + STEP_SIZE * l_g
            past_objective_value = objective
    return mu, L