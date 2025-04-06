import torch
from torch import nn


class Gate(nn.Module):

    def __init__(self, embedded_dim: int, hidden_dim: int, activation_function: nn.Module):
        super().__init__()
        self.hidden_w = nn.Parameter(torch.ones((hidden_dim, hidden_dim)))
        self.input_w = nn.Parameter(torch.ones((embedded_dim, hidden_dim)))
        self.b = nn.Parameter(torch.ones(hidden_dim))
        self.activation_function = activation_function

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = (torch.matmul(h, self.hidden_w) + torch.matmul(x, self.input_w)) + self.b
        return self.activation_function(z)