import torch
import math
import conv1d_cpp
from einops import rearrange

class conv1dFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, padding):
        outputs = conv1d_cpp.forward(input, weights, bias, padding)
        ctx.padding = padding
        ctx.save_for_backward(input, weights, bias)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        pass
    
class DepthWiseConv1d(torch.nn.Module):
    def __init__(self, channels, kernel_size, padding, weights, bias, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DepthWiseConv1d, self).__init__()
        self.d = channels
        self.k = kernel_size
        self.padding = padding
        self.weights  = torch.nn.Parameter(rearrange(weights.squeeze(), 'd k -> k d').detach().clone().contiguous())
        self.bias = torch.nn.Parameter(bias.detach().clone().contiguous())
        self.reset_parameters(weights, bias)

    def reset_parameters(self, weights, bias):
        pass

    def load_state_dict(self, state_dict, strict: bool = True):
        pass

    def save_state_dict(self):
        pass
    
    def forward(self, input):
        return conv1dFunc.apply(input, self.weights, self.bias, self.padding)