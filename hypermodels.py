import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD
import torch.nn.functional as F
from utils import generate_hypersphere


"""
Hypermodel contains
ready-to-use functions to update g and sample
"""
class Hypermodel(object):
    def __init__(self, observation_model_f, posterior_model_g, device, *args, **kwargs):
        self.g = posterior_model_g
        self.update_device(device)
        self.prior = self.g.sample_prior().to(self.device)
        self.f = observation_model_f

    def sample_posterior(self, n_samples):
        return self.g.sample_theta(n_samples) + self.prior # NEED TO DISCUSS THIS

    def sample_observations(self, data_point_x, n_samples):
        thetas = self.g.sample_theta(n_samples)
        return self.f(thetas, data_point_x)

    def update_device(self, device):
        self.device = device
        self.g.model = self.g.model.to(device)

    def update_g(self, data_loader, num_steps, num_z_samples, learning_rate, sigma_prior, sigma_obs, print_every=5):
        optimizer = SGD(self.g.model.parameters(), lr=learning_rate, weight_decay=0.)
        steps_done = 0
        total_loss = 0
        while True:
            for batch in data_loader:
                # x of size (batch, ) bandits or (batch, assortment_size) for assortment optimization
                # y of size (batch, ), floats
                # a of size (batch, index_size = m)
                x, y, a = batch
                x = x.to(self.device)
                y = y.to(self.device)
                a = a.to(self.device) # shape B, m

                # Sampling from the base distribution
                z_sample = self.g.model.sample_z(num_z_samples) # shape num_z_samples, K, m
                # Noisy observations
                if len(x.size() < 2):
                    z_sliced = torch.index_select(z_sample, 1, x) # shape M, B, m
                else:
                    z_sliced = torch.index_select(z_sample, 1, x.view(-1))
                    z_sliced = z_sliced.view(-1, x.size(0), x.size(1)) # shape M, B, m
                sigAz = sigma_obs * (a.unsqueeze(0) * z_sliced).sum(-1) #shape M, B
                # Hypermodel mapping
                difference_samples = self.g.model(z_sample) # of shape M, K
                # L2 regularization
                reg = torch.norm(difference_samples, p=2, dim=1)/(2 * sigma_prior ** 2)
                # Environment model mapping
                outputs = self.f(difference_samples + self.prior, x)
                
                loss = (((y + sigAz - outputs) ** 2).mean(1) + reg).mean(0) #/ (2 * sigmao ** 2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                steps_done += 1
                if (steps_done % print_every) == 0:
                    print(f"step {steps_done}, loss:{loss.item():2f}")
                if steps_done >= num_steps:
                    return total_loss / num_steps


# g takes Model as argument
# Model's got to be a nn.Module implementing:
# init_parameters()
# sample_z()
# sample_prior
class HypermodelG(object):
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.model.init_parameters()
    
    def sample_prior(self):
        with torch.no_grad():
            return self.model.sample_prior() 

    def sample_theta(self, number_of_thetas):
        with torch.no_grad():
            z = self.model.sample_z(number_of_thetas)
            return self.model(z)        


class LinearModuleBandits(nn.Module):
    def __init__(self, k_bandits, model_dim, prior_std):
        super(LinearModuleBandits, self).__init__()
        self.k = k_bandits
        self.m = model_dim
        self.C = nn.Parameter(torch.ones(size=(k_bandits, model_dim)))
        self.mu = nn.Parameter(torch.ones(k_bandits))
        self.prior_std = prior_std
    
    def init_parameters(self):
        mu_sampled = torch.randn(self.k) * 0.05
        c_sampled = torch.randn(self.k, self.m) * 0.05
        self.C.data = c_sampled
        self.mu.data = mu_sampled
    
    def forward(self, z):
        """
        z of size (batch, Kbandits, modelsize)
        theta of size (batch, Kbandits)
        """
        return (self.C * z).sum(-1) + self.mu
    
    def sample_z(self, n_samples):
        return torch.randn(n_samples, self.k, self.m)
    
    def sample_prior(self):
        D = np.random.randn(self.k) * self.prior_std# dim (k, k)
        B = generate_hypersphere(dim=self.m, n_samples=self.k, norm=2) # dim (k, m)
        z = np.random.randn(self.k, self.m) # dim (k, m)
        return torch.from_numpy(D * (B * z).sum(-1))


class LinearModuleAssortmentOpt(nn.Module):
    def __init__(self, model_size, index_size, prior_std):
        super(LinearModuleAssortmentOpt, self).__init__()
        self.k = model_size
        self.m = index_size
        self.layer = nn.Linear(in_features=index_size * model_size, out_features=model_size)
        self.prior_std = prior_std
    
    def init_parameters(self):
        mu_sampled = torch.randn(self.k) * 0.05
        c_sampled = torch.randn(self.k, self.k * self.m) * 0.05
        # import pdb;
        # pdb.set_trace()
        self.layer.weight.data = c_sampled
        self.layer.bias.data = mu_sampled
    
    def forward(self, z):
        """
        z of size (batch, model_size, index_size)
        theta of size (batch, model_size)
        """
        z = z.view(z.size(0), -1).contiguous()
        # import pdb;
        # pdb.set_trace()
        return F.sigmoid(self.layer(z))
    
    def sample_z(self, n_samples):
        return torch.randn(n_samples, self.k, self.m)
    
    def sample_prior(self):
        z = np.random.rand(self.k) # dim (k, m)
        return torch.from_numpy(z)