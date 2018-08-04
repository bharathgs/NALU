import torch

from math import sqrt
from torch import Tensor, exp, log, nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.functional import tanh, sigmoid, linear


class NeuralAccumulator(nn.Module):
    """Basic NAC unit implementation 
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
        self.W_ = Parameter(Tensor(outputs, inputs))
        self.M_ = Parameter(Tensor(outputs, inputs))
        self.W = Parameter(tanh(self.W_) * sigmoid(self.M_))
        xavier_uniform_(self.W_), xavier_uniform_(self.M_)
        self.register_parameter('bias', None)

    def forward(self, input):
        return linear(input, self.W, self.bias)
