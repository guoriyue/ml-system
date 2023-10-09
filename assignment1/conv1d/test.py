import torch 
import time
from torch import nn
from einops import rearrange
import random
from conv1d import DepthWiseConv1d
from prettytable import PrettyTable

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


results = PrettyTable()
results.field_names = ["B", "L", "D", "torch time (ms)", "cudatime (ms)", "speedup", "Effective bandwidth (GB/s)", "TFLOPS"]
for b in [1]:
    for l in [1024, 2048, 4096, 8192]:
        for d in [768, 1024, 2048, 8192]:
            x = torch.randn([b, l, d])
            
            conv1d_torch = nn.Conv1d(
                in_channels = d,
                out_channels = d,
                kernel_size = filter_length,
                groups = d,
                padding = padding
            )
            
            conv1d_cuda = DepthWiseConv1d(channels = d,
                                            kernel_size=filter_length,
                                            padding=padding,
                                            weights=conv1d_torch.weight,
                                            bias=conv1d_torch.bias
                                            )
            
            
            x_torch = rearrange(x, 'b l d -> b d l').contiguous()
            
            #warmup
            y_torch = conv1d_torch(x_torch)
            
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeats):
                y_torch = conv1d_torch(x_torch)
            torch.cuda.synchronize()
            torch_time = (time.time() - start)*1000/repeats
            
            y_cuda = conv1d_cuda(x)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeats):
                y_cuda = conv1d_cuda(x)
            torch.cuda.synchronize()
            cuda_time = (time.time() - start)*1000/repeats
            
            test_correctness(y_torch, rearrange(y_cuda, 'b d l -> b l d'))
            speedup = torch_time / cuda_time
            effective_bandwidth = (b * l * d * 2 * nbytes + filter_length * d * nbytes) / (cuda_time * 1e-3) / (2**30)
            l_out = l + 2 * padding - filter_length + 1
            tera_flops = (b * l_out * d * 2 * filter_length) / (cuda_time * 1e-3) / (2**40)
            results.add_row([b, l, d, torch_time, cuda_time, speedup, effective_bandwidth, tera_flops])
            
results.float_format = "0.2f"
print(results)