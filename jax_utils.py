import jax
from jax import jit, vmap
import jax.numpy as jnp


def solve_mixture_jax():
    return


def top_actions_factory(num_actions):
    def get_top_actions(posterior_sample):
        # posterior_sample of size (N,)
        return jnp.argsort(-posterior_sample)[:num_actions]

    final_get_top_actions = jit(vmap(get_top_actions, in_axes=0))
    # return type of shape (n_samples, K)
    return final_get_top_actions


def hash_actions_factory(hasher):
    def hash_action(action):
        return action.dot(hasher)

    final_hash_actions = jit(vmap(hash_action, in_axes=[0]))
    return final_hash_actions


def expected_rewards_a_star(a_star_ix, all_a_star_er, er):
    return jax.lax.dynamic_update_index_in_dim(
        all_a_star_er, er, a_star_ix, axis=0
    )


def expected_reward(posterior_sample, assortment):
    return jnp.sum(posterior_sample[assortment])


def flat_er():
    # return type of shape (n_samples,)
    return jit(vmap(expected_reward, in_axes=[0, 0]))


def all_er():
    # return type of shape (n_actions, n_samples)
    return jit(
        vmap(vmap(expected_reward, in_axes=[0, None]), in_axes=[None, 0])
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
            vmap(expected_rewards_a_star, in_axes=[0, None, 0])(
                row_to_a_star_ix, jnp.zeros(n_a_stars), ers_single_action
            ).sum(0)
            / all_a_star_counts
        )
        return (
            all_a_star_probas * (er_given_a_stars - mean_er_single_action) ** 2
        ).sum()

    # return type of shape (n_actions,)
    return jit(
        vmap(variance_single_action, in_axes=[None, None, None, None, 0, 0])
    )
