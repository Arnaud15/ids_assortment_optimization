from hypermodels import Hypermodel, HypermodelG, LinearModuleBandits
import torch
import numpy as np
from utils import generate_hypersphere
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

    def act(self):
        action = np.argmax(self.prior_belief)
        # action = np.random.randint(self.narms)
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
                                 reg_weight=self.params.reg_weight,
                                 sigma_obs=self.params.training_sigmaobs,
                                 step_t=len(self.dataset) + 1,
                                 print_every=self.params.printinterval if self.params.printinterval > 0 else self.params.nsteps + 1)
        self.prior_belief = self.hypermodel.sample_posterior(1).numpy().flatten()
        return reward

if __name__ == "__main__":
    from env import KBandits
    H = 100
    k = 5
    sigmao = 0.01 # environment parameters
    sigmap = 1.
    environment = KBandits(k=k, sigma_obs=sigmao, sigma_model=sigmap)
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--reg_weight", type=float, default=1.)
    parser.add_argument("--training_sigmaobs", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--model_input_dim", type=int, default=1)
    parser.add_argument("--nsteps", type=int, default=25)
    parser.add_argument("--printinterval", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--nzsamples", type=int, default=32)
    parser.add_argument("--prior_std", type=float, default=1.) #used only for the bandits setting
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
            #print(f"arm selected is {arm_selected}")
            reward = envnmt.step(arm_selected)
            obs[ix] = (arm_selected, reward)
            reward = actor.update(reward)  # agent observes item selected, perceive reward and updates its beliefs
            rewards[ix] = reward
            if (ix > n_steps - 2) or (ix % 10 == 0):
                data_test = actor.hypermodel.sample_posterior(1000)
                print(f"agent posterior sample: {data_test.mean(0)}, {data_test.std(0)}")
        # import pdb;
        # pdb.set_trace()
        from collections import Counter
        print(sorted([(key, i) for (key, i) in Counter([arm for (arm, rew) in obs]).items()], key=lambda x:x[0]))
        return obs, rewards
    myagent = ThompsonSamplingAgentBandits(k_bandits=k, params=args)
    run_episode(environment, actor=myagent, n_steps=H)
    print(f"environment arms = {environment.rewards}")