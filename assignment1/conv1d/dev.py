import torch 
import time
from torch import nn
from einops import rearrange
import random
from conv1d import DepthWiseConv1d

#correctness test
def test_correctness(x, y, atol=1e-2):
    assert torch.allclose(x, y, atol=atol), f"Expected {x} to equal {y}"

torch.manual_seed(229)
   
repeats = 15
filter_length = 3
padding = filter_length - 1

nbytes = 2
torch.set_default_dtype(torch.float16)
torch.set_default_device('cuda')

b = 2
l = 1024
d = 768

x = torch.randn([b, l, d])

x_torch = rearrange(x, 'b l d -> b d l').contiguous()
            

conv1d_torch = nn.Conv1d(
                in_channels = d,
                out_channels = d,
                kernel_size = filter_length,
                groups = d,
                padding = padding
            )

y_torch = conv1d_torch(x_torch)

conv1d_cuda = DepthWiseConv1d(channels = d,
                                kernel_size=filter_length,
                                padding=padding,
                                weights=conv1d_torch.weight,
                                bias=conv1d_torch.bias
                )

y_cuda = conv1d_cuda(x)


print((y_torch - rearrange(y_cuda, 'b d l -> b l d')).abs().max().item())
test_correctness(y_torch, rearrange(y_cuda, 'b d l -> b l d'))