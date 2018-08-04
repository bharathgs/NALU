import torch

from math import sqrt
from torch import Tensor, exp, log, nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.functional import tanh, sigmoid, linear
from NeuralAccumulator import NeuralAccumulator


class NALU(nn.Module):
    """Basic NALU unit implementation 
    from https://arxiv.org/pdf/1808.00508.pdf
    """

    def __init__(self, inputs, outputs):
        """
        inputs: input sample size
        outputs: output sample size
        """
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.G = Parameter(Tensor(outputs, inputs))
        self.W = Parameter(Tensor(outputs, inputs))
        self.nac = NeuralAccumulator(outputs, inputs)
        self.eps = 1e-5
        self.register_parameter('bias', None)
        xavier_uniform_(self.G), xavier_uniform_(self.W)

    def forward(self, input):
        a = self.nac(input)
        g = sigmoid(linear(input, self.G, self.bias))
        ag = g * a
        log_in = log(abs(input) + self.eps)
        m = exp(linear(log_in, self.W, self.bias))
        md = (1 - g) * m
        return ag + md
