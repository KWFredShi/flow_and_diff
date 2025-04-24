import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
# This is the correct import for the ODE solver that uses the adjoint method
# from torchdiffeq import odeint  
# Uncomment this if you want to use the standard ODE solver
# see https://github.com/rtqichen/torchdiffeq for more details


# Define the ODE Function
class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, dim)
        )
        # Initialize weights to small values
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, z):
        return self.net(z)

# CNF wrapper
class CNF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.odefunc = ODEFunc(dim)

    def forward(self, z0, t=torch.tensor([0.0, 1.0])):
        z_t = odeint(self.odefunc, z0, t, method='rk4')
        return z_t[1]  # Return the final state

    def inverse(self, z1, t=torch.tensor([1.0, 0.0])):
        z_t = odeint(self.odefunc, z1, t, method='rk4')
        return z_t[1]

# CNF VAE model
class CNFVAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim * 2)
        )

        self.flow = CNF(input_dim)

        self.decoder_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def encoder(self, x):
        h = self.encoder_net(x)
        mu, std = h.chunk(2, dim=1)
        std = F.softplus(std) + 1e-6
        return mu, std

    def decoder(self, z):
        return self.decoder_net(z)

    def forward(self, x):
        mu, std = self.encoder(x)
        eps = torch.randn_like(mu)
        z0 = mu + std * eps
        zk = self.flow(z0)
        x_recon = self.decoder(zk)
        return x_recon, mu, std

# VAE Loss (No log det term here, as CNF can use other techniques like trace estimation)
def cnf_vae_loss(x, x_recon, mu, std):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + std.log() - mu.pow(2) - std.pow(2))
    return recon_loss + kl
