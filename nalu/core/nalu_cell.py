from torch import Tensor, exp, log, nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.functional import sigmoid, linear
from .nac_cell import NacCell


class NaluCell(nn.Module):
    """Basic NALU unit implementation 
    from https://arxiv.org/pdf/1808.00508.pdf
    """

    def __init__(self, in_shape, out_shape):
        """
        in_shape: input sample dimension
        out_shape: output sample dimension
        """
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.G = Parameter(Tensor(out_shape, in_shape))
        self.nac = NacCell(out_shape, in_shape)
        xavier_uniform_(self.G)
        self.eps = 1e-5
        self.register_parameter('bias', None)

    def forward(self, input):
        a = self.nac(input)
        g = sigmoid(linear(input, self.G, self.bias))
        ag = g * a
        log_in = log(abs(input) + self.eps)
        m = exp(self.nac(log_in))
        md = (1 - g) * m
        return ag + md
