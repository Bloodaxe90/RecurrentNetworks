import torch
from torch import nn

from src.utils.gate import Gate


class LSTMLayer(nn.Module):

    def __init__(self, embedded_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.forget_gate = Gate(embedded_dim, hidden_dim, nn.Sigmoid())
        self.input_gate = Gate(embedded_dim, hidden_dim, nn.Sigmoid())
        self.candidate_layer = Gate(embedded_dim, hidden_dim, nn.Tanh())
        self.output_gate = Gate(embedded_dim, hidden_dim, nn.Sigmoid())
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = x.shape[:-1]
        device = x.device.type
        c_t = torch.zeros((batch_size, self.hidden_dim)).to(device)
        h_t = torch.zeros((batch_size, self.hidden_dim)).to(device)

        hidden_states: torch.Tensor = torch.Tensor()
        for step in range(seq_length):
            x_t = x[:, step, :]
            f_t = self.forget_gate(x_t, h_t)
            i_t = self.input_gate(x_t, h_t)
            candidate_c = self.candidate_layer(x_t, h_t)

            c_t = (f_t * c_t) + (i_t * candidate_c)
            o_t = self.output_gate(x_t, h_t)
            h_t = o_t * self.tanh(c_t)
            hidden_states = torch.cat((hidden_states, h_t.unsqueeze(1)), dim= -2)

        return hidden_states