import torch
import torch.nn as nn
import torch.nn.functional as F

# Coupling Layer (RealNVP)
class CouplingLayer(nn.Module):
    def __init__(self, dim, flip=False):
        super().__init__()
        self.dim = dim
        self.flip = flip
        self.scale_net = nn.Sequential(
            nn.Linear(self.dim // 2, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim // 2),
            nn.Tanh()
        )
        self.shift_net = nn.Sequential(
            nn.Linear(self.dim // 2, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim // 2)
        )

    def forward(self, x):
        if self.flip:
            x2, x1 = x.chunk(2, dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)

        s = self.scale_net(x1)
        t = self.shift_net(x1)
        y2 = x2 * torch.exp(s) + t

        if self.flip:
            y = torch.cat([y2, x1], dim=1)
        else:
            y = torch.cat([x1, y2], dim=1)

        log_det = s.sum(dim=1)
        return y, log_det

    def inverse(self, y):
        if self.flip:
            y2, y1 = y.chunk(2, dim=1)
        else:
            y1, y2 = y.chunk(2, dim=1)

        s = self.scale_net(y1)
        t = self.shift_net(y1)
        x2 = (y2 - t) * torch.exp(-s)

        if self.flip:
            x = torch.cat([x2, y1], dim=1)
        else:
            x = torch.cat([y1, x2], dim=1)

        return x


# Normalizing Flow
class NormalizingFlow(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            CouplingLayer(dim, flip=(i % 2 == 1)) for i in range(num_layers)
        ])

    def forward(self, z):
        log_det_total = 0
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det
        return z, log_det_total

    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

# Full VAE with Flow
class FlowVAE(nn.Module):
    def __init__(self, input_dim, num_flows=4):
        super().__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim * 2)
        )

        self.flow = NormalizingFlow(input_dim, num_flows)

        self.decoder_net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )

    def encoder(self, x):
        h = self.encoder_net(x)
        mu, std = h.chunk(2, dim=1)
        std = F.softplus(std) + 1e-6  # Ensure positivity
        return mu, std

    def decoder(self, z):
        return self.decoder_net(z)

    def forward(self, x):
        mu, std = self.encoder(x)
        eps = torch.randn_like(mu)
        z0 = mu + std * eps
        zk, log_det = self.flow(z0)
        x_recon = self.decoder(zk)
        return x_recon, mu, std, log_det


def flow_vae_loss(x, x_recon, mu, std, log_det):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + std.log() - mu.pow(2) - std.pow(2))
    return recon_loss + kl - log_det.sum()

