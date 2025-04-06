import torch
from torch import nn

from src.utils.gate import Gate


class GRULayer(nn.Module):

    def __init__(self, embedded_dim: int, hidden_dim: int, reset_first: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reset_first = reset_first

        self.reset_gate = Gate(embedded_dim, hidden_dim, nn.Sigmoid())
        self.update_gate = Gate(embedded_dim, hidden_dim, nn.Sigmoid())

        self.hidden_w = nn.Parameter(torch.ones((hidden_dim, hidden_dim)))
        self.input_w = nn.Parameter(torch.ones((embedded_dim, hidden_dim)))
        self.bias = nn.Parameter(torch.ones(hidden_dim))
        self.tanh = nn.Tanh()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = x.shape[:-1]
        h_t = torch.zeros((batch_size, self.hidden_dim)).to(x.device.type)

        hidden_states: torch.Tensor = torch.Tensor()
        for step in range(seq_length):
            x_t = x[:, step, :]
            r_t = self.reset_gate(x_t, h_t)
            z_t = self.update_gate(x_t, h_t)

            candidate_h = self.tanh((r_t * torch.matmul(h_t, self.hidden_w)) + torch.matmul(x_t, self.input_w) + self.bias)

            h_t = (z_t * h_t) + ((1 - z_t) * candidate_h)
            hidden_states = torch.cat((hidden_states, h_t.unsqueeze(1)), dim= -2)

        return hidden_states