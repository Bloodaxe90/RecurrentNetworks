import torch
from torch import nn


class RecurrentLayer(nn.Module):

    def __init__(self, embedded_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.hidden_w: torch.Tensor = nn.Parameter(torch.ones(hidden_dim, hidden_dim))
        self.input_w: torch.Tensor = nn.Parameter(torch.ones(embedded_dim, hidden_dim))
        self.b: torch.Tensor = nn.Parameter(torch.ones(hidden_dim))
        self.tanh = nn.Tanh()

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = x_seq.shape[:-1]
        hidden_state = torch.zeros(batch_size, self.hidden_dim).to(x_seq.device.type)

        hidden_states = torch.Tensor()
        for step in range(seq_length):
            z = torch.matmul(hidden_state, self.hidden_w) + torch.matmul(x_seq[..., step, :], self.input_w) + self.b
            hidden_state = self.tanh(z)
            hidden_states = torch.cat((hidden_states, hidden_state.unsqueeze(1)), dim= -2)
        return hidden_states