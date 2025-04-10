import torch
from torch import nn


class RecurrentLayer(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.hidden_w = nn.Parameter(
            nn.init.orthogonal_(torch.empty(hidden_dim, hidden_dim))
        )
        self.input_w = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(input_dim, hidden_dim))
        )
        self.b: torch.Tensor = nn.Parameter(torch.zeros(hidden_dim))
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = x.shape[:2]
        hidden_state = torch.zeros(batch_size, self.hidden_dim).to(x.device.type)

        hidden_states = torch.Tensor().to(x.device.type)
        for step in range(seq_length):
            z = (
                torch.matmul(hidden_state, self.hidden_w)
                + torch.matmul(x[..., step, :], self.input_w)
                + self.b
            )
            hidden_state = self.tanh(z)
            hidden_states = torch.cat(
                (hidden_states, hidden_state.unsqueeze(1)), dim=-2
            )
        return hidden_states
