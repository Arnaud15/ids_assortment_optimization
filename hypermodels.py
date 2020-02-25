import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD, Adam
from utils import generate_hypersphere


class Hypermodel(object):
    def __init__(self, observation_model_f, posterior_model_g, device, *args, **kwargs):
        self.g = posterior_model_g
        self.update_device(device)
        self.f = observation_model_f

    def sample_posterior(self, n_samples):
        return self.g.sample_theta(n_samples)

    def sample_observations(self, data_point_x, n_samples):
        thetas = self.g.sample_theta(n_samples)
        return self.f(thetas, data_point_x)

    def update_device(self, device):
        self.device = device
        self.g.model = self.g.model.to(device)

    def update_g(self, data_loader, num_steps, num_z_samples, learning_rate, reg_weight, sigma_obs, step_t=1, print_every=5):
        optimizer = Adam(self.g.model.parameters(), lr=learning_rate)#SGD(self.g.model.parameters(), lr=learning_rate)
        steps_done = 0
        total_loss = 0
        while True:
            for batch_ix, batch in enumerate(data_loader):
                # x of size (batch, ) bandits or (batch, assortment_size) for assortment optimization
                # y of size (batch, ), floats
                # a of size (batch, index_size = m)
                x, y, a = batch
                if batch_ix > 0: #TODO remember this
                    break
                x = x.to(self.device)
                y = y.to(self.device)
                a = a.to(self.device) # shape B, m; B, m * assortment_size for assortment opt
                # Sampling from the base distribution
                z_sample = self.g.model.sample_z(num_z_samples) # shape num_z_samples, K, m
                # Noisy observations
                if len(x.size()) < 2:
                    z_sliced = torch.index_select(z_sample, 1, x) # shape M, B, m
                else:
                    z_sliced = torch.index_select(z_sample, 1, x.view(-1)) # shape M, B * assortment_size, m)
                    z_sliced = z_sliced.view(num_z_samples, x.size(0), -1) # shape (M, B, assortmen_size * m)
                sigAz = sigma_obs * (a.unsqueeze(0) * z_sliced).sum(-1) #shape M, B
                # Hypermodel mapping
                difference_samples = self.g.model(z_sample) # of shape M, K
                # L2 regularization
                reg = torch.norm(difference_samples, p=2, dim=1) * (reg_weight / step_t)
                # Environment model mapping
                outputs = self.f(difference_samples + self.g.model.prior(z_sample), x)
                
                loss = (((y + sigAz - outputs) ** 2).mean(1) + reg).mean(0) #/ (2 * sigmao ** 2) #TODO discuss the sum and the no sigmao denominator
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                steps_done += 1
                if (steps_done % print_every) == 0:
                    print(f"step {steps_done}, total loss:{loss.item():.3f}, reg: {reg.mean(0).item():.3f}")
                if steps_done >= num_steps:
                    return total_loss / num_steps


class HypermodelG(object):
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.model.init_parameters()
    
    def sample_theta(self, number_of_thetas):
        with torch.no_grad():
            z = self.model.sample_z(number_of_thetas)
            return self.model(z) + self.model.prior(z)


class LinearModuleBandits(nn.Module):
    def __init__(self, k_bandits, model_dim, prior_std=1.5):
        super(LinearModuleBandits, self).__init__()
        self.k = k_bandits
        self.m = model_dim
        self.C = nn.Parameter(torch.ones(size=(k_bandits, model_dim)))
        self.mu = nn.Parameter(torch.ones(k_bandits))
        self.D = torch.randn(self.k) * prior_std# dim (k, k)
        self.B = torch.from_numpy(generate_hypersphere(dim=self.m, n_samples=self.k, norm=2)).unsqueeze(0)
    
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
    
    def prior(self, z):
        """
        z of shape (B, k, m)
        """
        return self.D * (self.B * z).sum(-1)


class LinearModuleAssortmentOpt(nn.Module): 
    def __init__(self, model_size, index_size):
        super(LinearModuleAssortmentOpt, self).__init__()
        self.k = model_size
        self.m = index_size
        self.in_layer = nn.Linear(in_features=index_size * model_size, out_features=model_size)
        self.D = torch.randn(self.k) * 1.5 # 1.5 gives something ~close to an uniform[0, 1]
        self.B = torch.from_numpy(generate_hypersphere(dim=self.m, n_samples=self.k, norm=2)).unsqueeze(0)
    
    def init_parameters(self):
        # mu_sampled = torch.randn(self.k) * 0.05
        # c_sampled = torch.randn(self.k, self.k * self.m) * 0.05
        # import pdb;
        # pdb.set_trace()
        self.in_layer.weight.data = 0.05 * torch.randn(self.k, self.k * self.m)
        self.in_layer.bias.data.fill_(0.)
    
    def forward(self, z):
        """
        z of size (batch, model_size, index_size)
        theta of size (batch, model_size)
        """
        z = z.view(z.size(0), -1).contiguous()
        return self.in_layer(z) # torch.sigmoid(self.layer(z)) # TODO remember the need for a sigmoid in the ids agent
    
    def sample_z(self, n_samples):
        return torch.randn(n_samples, self.k, self.m)
    
    def prior(self, z): #TODO unclear if gonna work
        return self.D * (self.B * z).sum(-1)


class NeuralModuleAssortmentOpt(nn.Module):
    def __init__(self, model_size, index_size):
        super(NeuralModuleAssortmentOpt, self).__init__()
        self.k = model_size
        self.m = index_size
        self.in_layer = nn.Linear(in_features=index_size * model_size, out_features=model_size)
        self.activation = nn.ReLU()
        self.mid_layer = nn.Linear(in_features=model_size, out_features=model_size)
        self.activation2 = nn.ReLU()
        self.out_layer = nn.Linear(in_features=model_size, out_features=model_size)
        self.D = torch.randn(self.k) * 1.5 # 1.5 gives something ~close to an uniform[0, 1]
        self.B = torch.from_numpy(generate_hypersphere(dim=self.m, n_samples=self.k, norm=2)).unsqueeze(0)
    
    def init_parameters(self):
        modules = [self.in_layer, self.mid_layer, self.out_layer]
        for layer in modules:
            layer.weight.data = 0.05 * nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
            layer.bias.data.fill_(0.)
    
    def forward(self, z):
        """
        z of size (batch, model_size, index_size)
        theta of size (batch, model_size)
        """
        z = z.view(z.size(0), -1).contiguous()
        z = self.in_layer(z)
        z = self.activation(z)
        z = self.mid_layer(z)
        z = self.activation2(z)
        z = self.out_layer(z)
        return z # torch.sigmoid(self.layer(z)) # TODO remember the need for a sigmoid in the ids agent
    
    def sample_z(self, n_samples):
        return torch.randn(n_samples, self.k, self.m)
    
    def prior(self, z): #TODO unclear if gonna work
        return self.D * (self.B * z).sum(-1)

# TODO devices