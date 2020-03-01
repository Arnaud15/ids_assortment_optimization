import numpy as np
import abc
from scipy.stats import beta
from collections import defaultdict
from hypermodels import NeuralModuleAssortmentOpt, LinearModuleAssortmentOpt, Hypermodel, HypermodelG
from utils import generate_hypersphere
from torch.utils.data import DataLoader
import torch

SELECTED_HYPERMODEL = LinearModuleAssortmentOpt

class Agent(abc.ABC):
    def __init__(self, k, n):
        self.assortment_size = k
        self.n_items = n

    @abc.abstractmethod
    def act(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def sample_from_posterior(self, n_samples):
        """
        input is number of samples
        output of shape (n_samples, n_items) are the samples beliefs in (0, 1) for each sample, item
        """
        pass

    def perceive_reward(self, item):
        """
        :param item:  index in [0, n-1], "no item" is index n
        :return: reward of 1. if any item is selected, 0. otherwise
        """
        return 1. if item < self.n_items else 0.

    @abc.abstractmethod
    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        return reward


class RandomAgent(Agent):
    def __init__(self, k, n, **kwargs):
        super().__init__(k, n)

    def act(self):
        return np.random.choice(np.arange(self.n_items, dtype=int), size=self.assortment_size, replace=False)

    def reset(self):
        pass

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        return reward

    def sample_from_posterior(self, n_samples):
        return np.random.rand(n_samples, self.n_items)


class EpochSamplingAgent(Agent, abc.ABC):
    def __init__(self, k, n, horizon=None, correlated_sampling=False):
        Agent.__init__(self, k, n)
        self.correlated_sampling = correlated_sampling
        self.T = horizon
        self.reset()

    def act(self):
        if self.epoch_ended:
            action = self.proposal()
        else:
            action = self.current_action
        return action

    def sample_from_posterior(self, n_samples):
        if not self.correlated_sampling:
            return np.array(
                [(1 / beta.rvs(a=a_, b=b_, size=n_samples)) - 1 for (a_, b_) in self.posterior_parameters]).T
        else:
            gaussian_means = np.expand_dims([b_ / a_ for (a_, b_) in self.posterior_parameters], 0)
            gaussian_stds = np.expand_dims(
                [np.sqrt((b_ / a_) * ((b_ / a_) + 1) / a_) for
                 (a_, b_) in self.posterior_parameters], 0)
            # print(gaussian_means, gaussian_stds)
#             import ipdb;
#             ipdb.set_trace()
            theta_sampled = np.random.randn(n_samples, 1)
            sample = gaussian_means + theta_sampled * gaussian_stds
            # print(f"sampled posterior parameters are: {sample}")
            return sample

    def reset(self):
        self.epoch_ended = True
        self.current_action = self.n_items
        self.epoch_picks = defaultdict(int)
        self.posterior_parameters = [(1, 1) for _ in range(self.n_items)]

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        if item_selected == self.n_items:
            self.epoch_ended = True
            # print(f"former posterior parameters where: {self.posterior_parameters}")
            n_is = [int(ix in self.current_action) for ix in range(self.n_items)]
            # print("current action", self.current_action)
            # print("nis", n_is)
            v_is = [self.epoch_picks[i] for i in range(self.n_items)]
            # print("epoch picks", self.epoch_picks)
            # print("vis", v_is)
            self.posterior_parameters = [(a + n_is[ix], b + v_is[ix]) for ix, (a, b) in
                                            enumerate(self.posterior_parameters)]
            # print(f"Now they are {self.posterior_parameters}")
            self.epoch_picks = defaultdict(int)
        else:
            self.epoch_picks[item_selected] += 1
            self.epoch_ended = False
        return reward

    @abc.abstractmethod
    def proposal(self):
        pass

def f_assortment_optimization(thetas, x):
    """
    thetas: size (n_z_sampled, model_size) UNNORMALIZED
    x: size (Batch, assortment_size) long tensor
    """
    thetas = torch.sigmoid(thetas)
    thetas_selected = torch.index_select(thetas, 1, x.view(-1)) # size (n_z_sampled, batch * assort_size)
    thetas_selected = thetas_selected.view(-1, x.size(0), x.size(1)) #size (n_z_sampled, batch, assort_size)
    thetas_sum = thetas_selected.sum(-1)
    return thetas_sum / (1 + thetas_sum) # size (n_z, batch)


class HypermodelAgent(Agent, abc.ABC):
    def __init__(self, k, n, params, n_samples=1):
        Agent.__init__(self, k, n)
        self.params = params
        self.n_samples = n_samples
        self.reset()

    def sample_from_posterior(self, nsamples=1):
        return torch.sigmoid(self.hypermodel.sample_posterior(nsamples)).numpy()

    def reset(self):
        linear_hypermodel = SELECTED_HYPERMODEL(model_size=self.n_items,
                                                index_size=self.params.model_input_dim)
        g_model = HypermodelG(linear_hypermodel)
        self.hypermodel = Hypermodel(observation_model_f=f_assortment_optimization, 
                                     posterior_model_g=g_model,
                                     device='cpu')
        self.prior_belief = self.sample_from_posterior(self.n_samples)
        self.current_action = self.n_items
        self.dataset = []

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected) 
        data_point = [self.current_action, reward, generate_hypersphere(dim=self.params.model_input_dim * self.assortment_size,
                                                                        n_samples=1,
                                                                        norm=2)[0]] 
        self.dataset.append(data_point)
        data_loader = DataLoader(self.dataset, batch_size=self.params.batch_size, shuffle=True)
        self.hypermodel.update_g(data_loader,
                                 num_steps=self.params.nsteps,
                                 num_z_samples=self.params.nzsamples,
                                 learning_rate=self.params.lr,
                                 reg_weight=self.params.reg_weight,
                                 sigma_obs=self.params.training_sigmaobs,
                                 step_t=len(self.dataset) + 1,
                                 print_every=self.params.printinterval if self.params.printinterval > 0 else self.params.nsteps + 1)
        self.prior_belief = self.sample_from_posterior(self.n_samples)
        return reward
    
    def act(self):
        raise NotImplementedError