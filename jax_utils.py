import jax
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
import logging

DISCRETIZATION_IDS = jnp.linspace(0.0, 1.0, num=15)[:-1]


@jit
def info_ratio(d1, d2, v1, v2, rho):
    delta_2 = (d1 * rho + d2 * (1.0 - rho)) ** 2
    gain = v1 * rho + (1.0 - rho) * v2
    return delta_2 / gain


info_ratio_mapped = vmap(
    vmap(
        vmap(info_ratio, in_axes=[None, None, None, None, 0]),
        in_axes=[None, 0, None, 0, None],
    ),
    in_axes=[0, None, 0, None, None],
)


@jit
def flat_ix_to_a1_a2_rho(flat_ix, n_actions, discretization_size):
    ix = flat_ix // (n_actions * discretization_size)
    flat_ix_2 = flat_ix - ix * n_actions * discretization_size
    iy = flat_ix_2 // discretization_size
    iz = flat_ix_2 % discretization_size
    return ix, iy, iz


def solve_mixture_jax(regrets, variances):
    idx_flat = info_ratio_mapped(
        regrets, regrets, variances, variances, DISCRETIZATION_IDS
    ).argmin()
    (ix, iy, iz) = flat_ix_to_a1_a2_rho(
        idx_flat,
        n_actions=regrets.shape[0],
        discretization_size=DISCRETIZATION_IDS.shape[0],
    )
    logging.info(f"{ix}, {iy}, {iz}")
    logging.info(f"rho: {DISCRETIZATION_IDS[iz]:.2f}")
    logging.info(f"a1(r, g): {regrets[ix]:.2f}, {variances[ix]:.2f}")
    logging.info(f"a2(r, g): {regrets[iy]:.2f}, {variances[iy]:.2f}")
    if np.random.rand() <= DISCRETIZATION_IDS[iz]:
        return ix
    else:
        return iy


def top_actions_factory(num_actions):
    def get_top_actions(posterior_sample):
        # posterior_sample of size (N,)
        return jnp.argsort(-posterior_sample)[:num_actions]

    final_get_top_actions = vmap(jit(get_top_actions), in_axes=0)
    # return type of shape (n_samples, K)
    return final_get_top_actions


def hash_actions_factory(hasher):
    def hash_action(action):
        return action.dot(hasher)

    final_hash_actions = vmap(jit(hash_action), in_axes=[0])
    return final_hash_actions


def expected_rewards_a_star(a_star_ix, all_a_star_er, er):
    return jax.lax.dynamic_update_index_in_dim(
        all_a_star_er, er, a_star_ix, axis=0
    )


def expected_reward(posterior_sample, assortment):
    return jnp.sum(posterior_sample[assortment])


def flat_er():
    # return type of shape (n_samples,)
    return vmap(jit(expected_reward), in_axes=[0, 0])


def all_er():
    # return type of shape (n_actions, n_samples)
    return vmap(
        vmap(jit(expected_reward), in_axes=[0, None]), in_axes=[None, 0]
    )


def variances_factory():
    def variance_single_action(
        row_to_a_star_ix,
        n_a_stars,
        all_a_star_counts,
        all_a_star_probas,
        ers_single_action,
        mean_er_single_action,
    ):
        er_given_a_stars = (
            vmap(jit(expected_rewards_a_star), in_axes=[0, None, 0])(
                row_to_a_star_ix, jnp.zeros(n_a_stars), ers_single_action
            ).sum(0)
            / all_a_star_counts
        )
        return (
            all_a_star_probas * (er_given_a_stars - mean_er_single_action) ** 2
        ).sum()

    # return type of shape (n_actions,)
    return vmap(variance_single_action, in_axes=[None, None, None, None, 0, 0])
