from torch.nn import Sequential
from torch import nn
from nalu.core.nalu_cell import NaluCell


class NaluLayer(nn.Module):
    def __init__(self, input_shape, output_shape, n_layers, hidden_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_layers = n_layers
        self.hidden_shape = hidden_shape
        layers = [NaluCell(hidden_shape if n > 0 else input_shape,
                           hidden_shape if n < n_layers - 1 else output_shape) for n in range(n_layers)]
        self.model = Sequential(*layers)

    def forward(self, data):
        return self.model(data)
