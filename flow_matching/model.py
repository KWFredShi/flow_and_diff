import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class flow_matching_OT_displacement(nn.Module):
    """
    Flow Matching with Optimal Transport Displacement
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = self.input_dim
        self.flow_predictor = nn.Sequential(
            nn.Linear(input_dim + self.time_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim)
        )

    def time_embedding(self, t, max_timescale=1e4):
        half = self.time_dim // 2
        scales = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * -(math.log(max_timescale) / (half-1))
        )  # shape [half]
        t = t.view(-1,1)
        args = t * scales[None,:]          # [B,half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B,emb_dim]

    
    def forward(self, x, t):
        """t
        Forward pass for flow matching
        """
        x_t = torch.cat([x, self.time_embedding(t)], dim=1)
        flow = self.flow_predictor(x_t)
        return flow
    
def OT_displacement_cond_loss(model, x0, x1, t, sigma_min):
    """
    Loss function for flow matching
    """
    u_t = (x1 - (1 - sigma_min) * x0)
    phi_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
    loss = F.mse_loss(u_t, model(phi_t, t))
    return loss

