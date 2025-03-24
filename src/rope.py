import torch
import torch.nn as nn


class RoPEPositionalEncoding(nn.Module):
    """
    Classic rope take as input the positions (for instance tensor(1,2,3,4 ...)) and return
    the rope embedding, so of dimension [len(tensor), d_model], as a tensor
    """

    def __init__(self, d_model, max_len=1000, base=100, device="cuda"):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base  # base is the value of theta since theta_i = theta.pow(-2*i/d_model)
        self.theta = torch.tensor([base ** (-2 * (i // 2) / d_model) for i in range(d_model)]).to(device)
        self.device = device
        self.to(device)

    def forward(self, positions):
        angles = positions.unsqueeze(-1) * self.theta
        sin_cos = torch.stack([angles.cos(), angles.sin()], dim=-1)
        return sin_cos.view(*sin_cos.shape[:-2], -1)  # [B, input_length, 2*d_model]

    # for instance if u want to plot the embedding in a 2d graph, simply do:
    # rope = RoPEPositionalEncoding(1)
    # out = rope(torch.tensor(np.arange(n)))
    # out is [n x 2], more generally you multiply by two the d_model
