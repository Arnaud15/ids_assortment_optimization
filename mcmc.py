import pymc3 as pm
import theano.tensor as tt

# import theano.printing as pt  # for debugging purposes
import logging
import warnings

logger = logging.getLogger("pymc3")
logger.setLevel(logging.ERROR)


def sample_from_posterior(
    n_samples, assortments, item_picks, n_observations, n_items
):
    """
    :param n_samples: number of independent samples to draw from the posterior
    :param assortments: actions history, ndarray of shape (n_obs, N) of one hot encoding (each row sums to K)
    :param item_picks: items selected by the buyer, ndarray of shape (n_obs,) of ints in [0, ..., N] (N == no item selected)
    :param n_observations
    :return: ndarray of shape (n_samples, N) = independant samples from the posterior distribution
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pm.Model() as glm_model:
            v = pm.Uniform("v", lower=0.0, upper=1.0, shape=n_items)
            v = tt.concatenate((v, tt.as_tensor([1.0])))
            mask = pm.Deterministic("assortments", tt.as_tensor(assortments))
            v_offered = mask * v
            which_item = pm.Categorical(
                "which_item",
                p=v_offered / tt.sum(v_offered, axis=1, keepdims=True),
                shape=n_observations,
                observed=item_picks,
            )
            trace_full = pm.sample(
                1, tune=250, chains=n_samples, progressbar=False
            )
    return trace_full["v"]

    #     v_printed = pt.Print('vector', attrs = [ 'shape' ])()
    #     v_printed = pt.Print('vector')(tt.concatenate(v, tt.as_tensor([1.])))
    #     pick_no_item = pm.Bernoulli('no_item', p=1/(1+tt.sum(v_of_offered, axis=1)), shape=nsim)
    #     v_printed = pt.Print('vector')(v_of_offered/tt.sum(v_of_offered, axis=1, keepdims=True))
