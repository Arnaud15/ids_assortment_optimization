from mcmc import sample_from_posterior
from scipy.stats import uniform
from utils import act_optimally, generate_hypersphere
from base_agents import Agent, EpochSamplingAgent, HypermodelAgent
import numpy as np
from collections import defaultdict


class ThompsonSamplingAgent(Agent):
    def __init__(self, k, n, **kwargs):
        super().__init__(k, n)
        self.prior_belief = uniform.rvs(size=n)
        self.assortments_given = []
        self.item_picks = []

    def act(self):
        action = act_optimally(self.prior_belief, top_k=self.assortment_size)
        assortment = np.zeros(self.n_items + 1)
        assortment[self.n_items] = 1.
        for item in action:
            assortment[item] = 1.
        self.assortments_given.append(assortment)
        return action

    def reset(self):
        self.prior_belief = uniform.rvs(size=self.n_items)
        self.assortments_given = []
        self.item_picks = []

    def update(self, item_selected):
        reward = self.perceive_reward(item_selected)
        self.item_picks.append(item_selected)
        self.prior_belief = np.squeeze(sample_from_posterior(n_samples=1,
                                                             assortments=np.array(self.assortments_given),
                                                             item_picks=np.array(self.item_picks),
                                                             n_observations=len(self.item_picks),
                                                             n_items=self.n_items))
        return reward


class EpochSamplingTS(EpochSamplingAgent):
    def __init__(self, k, n, horizon, correlated_sampling, **kwargs):
        super().__init__(k, n, horizon=horizon, correlated_sampling=correlated_sampling)

    def proposal(self):
        posterior_belief = self.sample_from_posterior(1)
        action = act_optimally(np.squeeze(posterior_belief), top_k=self.assortment_size)
        self.current_action = action
        return action

    def reset(self):
        self.epoch_ended = True
        self.current_action = self.n_items
        self.epoch_picks = defaultdict(int)
        self.posterior_parameters = [(1, 1) for _ in range(self.n_items)]


class HypermodelTS(HypermodelAgent):
    def __init__(self, k, n, params, **kwargs):
        super().__init__(k, n, params, n_samples=1)

    def act(self):
        # action = act_optimally(np.squeeze(self.prior_belief), top_k=self.assortment_size)
        action = np.random.choice(np.arange(self.n_items, dtype=int), size=self.assortment_size, replace=False)
        self.current_action = action
        return action


from hypermodels import Hypermodel, HypermodelG, LinearModuleBandits
import torch
from torch.utils.data import DataLoader

def f_bandits(thetas, x):
    return torch.index_select(thetas, 1, x)

class ThompsonSamplingAgentBandits(object):
    def __init__(self, k_bandits, params): #TODO switch to assortment opt
        self.dataset = []
        self.narms = k_bandits
        self.params = params 
        linear_hypermodel = LinearModuleBandits(k_bandits=self.narms,
                                                model_dim=self.params.model_input_dim,
                                                prior_std=self.params.prior_std)
        g_model = HypermodelG(linear_hypermodel)
        self.hypermodel = Hypermodel(observation_model_f=f_bandits, 
                                     posterior_model_g=g_model,
                                     device='cpu')
        self.prior_belief = self.hypermodel.sample_posterior(1).numpy().flatten()
        self.current_action = None
        self.dataset = []

    def act(self):
        action = np.argmax(self.prior_belief)
        self.current_action = action
        return action

    def reset(self):
        linear_hypermodel = LinearModuleBandits(k_bandits=self.narms,
                                                model_dim=self.params.model_input_dim,
                                                prior_std=self.params.prior_std)
        g_model = HypermodelG(linear_hypermodel)
        self.hypermodel = Hypermodel(observation_model_f=f_bandits, 
                                     posterior_model_g=g_model,
                                     device='cpu')
        self.prior_belief = self.hypermodel.sample_posterior(1).numpy().flatten()
        self.current_action = None
        self.dataset = []

    def update(self, reward):
        data_point = [self.current_action, reward, generate_hypersphere(dim=self.params.model_input_dim,
                                                                        n_samples=1,
                                                                        norm=2)[0]] 
        self.dataset.append(data_point)
        data_loader = DataLoader(self.dataset, batch_size=self.params.batch_size, shuffle=True)
        self.hypermodel.update_g(data_loader,
                                 num_steps=self.params.nsteps,
                                 num_z_samples=self.params.nzsamples,
                                 learning_rate=self.params.lr,
                                 sigma_prior=self.params.training_sigmap,
                                 sigma_obs=self.params.training_sigmaobs,
                                 true_batch_size=self.params.batch_size,
                                 print_every=self.params.printinterval if self.params.printinterval > 0 else self.params.nsteps + 1)
        self.prior_belief = self.hypermodel.sample_posterior(1).numpy().flatten()
        return reward

if __name__ == "__main__":
    from env import KBandits
    H = 30
    k = 5
    sigmao = 0.01 # environment parameters
    sigmap = 1.
    environment = KBandits(k=k, sigma_obs=sigmao, sigma_model=sigmap)
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--training_sigmap", type=float, default=0.5)
    parser.add_argument("--training_sigmaobs", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_input_dim", type=int, default=3)
    parser.add_argument("--nsteps", type=int, default=25)
    parser.add_argument("--printinterval", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--nzsamples", type=int, default=32)
    parser.add_argument("--prior_std", type=float, default=0.5) #used only for the bandits setting
    args = parser.parse_args()


    def run_episode(envnmt, actor, n_steps):
        """
        :param envnmt: instance from the AssortmentEnvironment class
        :param actor: instance from the Agent class
        :param n_steps: horizon of the run in the environment
        :return: (observations history = list of (assortment one-hot array of size N+1, 0<=index<=N of item picked),
        rewards = historical rewards)
        """
        actor.reset()  # Resets the internal state of the agent at the start of simulations (prior beliefs, etc...)
        rewards = np.zeros(n_steps)
        obs = [0] * n_steps

        for ix in range(n_steps):
            arm_selected = actor.act()
            print(f"arm selected is {arm_selected}")
            reward = envnmt.step(arm_selected)
            obs[ix] = (arm_selected, reward)
            reward = actor.update(reward)  # agent observes item selected, perceive reward and updates its beliefs
            rewards[ix] = reward
            if ix > n_steps - 150:
                data_test = actor.hypermodel.sample_posterior(1000)
                print(f"agent posterior sample: {data_test.mean(0)}, {data_test.std(0)}")
        return obs, rewards
    myagent = ThompsonSamplingAgentBandits(k_bandits=k, params=args)
    run_episode(environment, actor=myagent, n_steps=H)
    print(f"environment arms = {environment.rewards}")