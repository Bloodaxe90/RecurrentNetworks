from torch import nn
import torch


class GeneralRNN(nn.Module):

    def __init__(
        self,
        recurrent_layer,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_recurrent_layers: int,
        dropout_prob: float = 0.2,
    ):
        super().__init__()
        assert num_recurrent_layers > 0, "Model must have at least one recurrent layer"
        self.recurrent_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    recurrent_layer((hidden_dim if i > 0 else input_dim), hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_prob),
                )
                for i in range(num_recurrent_layers)
            ]
        )
        self.linear_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent_x = x
        for idx, recurrent_block in enumerate(self.recurrent_blocks):
            recurrent_x = recurrent_block(recurrent_x)
        return self.linear_layer(recurrent_x)
