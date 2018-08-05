from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.functional import tanh, sigmoid, linear


class NacCell(nn.Module):
    """Basic NAC unit implementation 
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
        self.W_ = Parameter(Tensor(out_shape, in_shape))
        self.M_ = Parameter(Tensor(out_shape, in_shape))
        self.W = Parameter(tanh(self.W_) * sigmoid(self.M_))
        xavier_uniform_(self.W_), xavier_uniform_(self.M_)
        self.register_parameter('bias', None)

    def forward(self, input):
        return linear(input, self.W, self.bias)
