import torch
from torch import nn


class Gate(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, activation_function: nn.Module):
        super().__init__()
        self.hidden_w = nn.Parameter(
            nn.init.orthogonal_(torch.empty(hidden_dim, hidden_dim))
        )
        self.input_w = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(input_dim, hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.activation_function = activation_function

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = (torch.matmul(h, self.hidden_w) + torch.matmul(x, self.input_w)) + self.b
        return self.activation_function(z)
