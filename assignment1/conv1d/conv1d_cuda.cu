// Simple 1D depthwise convolution implementation with dilation and stride = 1

#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cuda_fp16.h>

const uint STRIDE = 1;
const uint DILATION = 1;

__global__ void conv1d_kernel(
    const __half2 *__restrict__ u,
    const __half2 *__restrict__ weights,
    const __half2 *__restrict__ bias,
    __half2 *__restrict__ out,
    uint padding,
    uint B,
    uint L,
    uint D,
    uint L_out,
    uint K)
{
    //TODO: implement a simple 1D depthwise convolution with dilation and stride = 1
}


//Do not change the function signature or return type!
torch::Tensor conv1d_cuda_half(
    torch::Tensor u,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding)
{
    const uint b = u.size(0);
    const uint l = u.size(1);
    const uint d = u.size(2);

    const uint k = weight.size(0);

    //block dimensions
    dim3 blockDims;

    //TODO: set the block dimensions
    // blockDims.x = ?
    // blockDims.y = ?
    // blockDims.z = ?

    dim3 gridDims;

    //TODO: set the grid dimensions
    // gridDims.x = ?
    // gridDims.y = ?
    // gridDims.z = ?


    //computes the output length. For more info see https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    uint l_out = ((l + 2 * padding - DILATION * (k - 1) - 1) / STRIDE + 1);

    //create the output tensor
    torch::Tensor out = torch::empty({b, l_out, d}, u.options());

    //feel free to modify the kernel call however you like
    conv1d_kernel<<<gridDims, blockDims>>>(
        static_cast<__half2 *>(u.data_ptr()),
        static_cast<__half2 *>(weight.data_ptr()),
        static_cast<__half2 *>(bias.data_ptr()),
        static_cast<__half2 *>(out.data_ptr()),
        padding,
        b,
        l,
        d,
        l_out,
        k);

    return out;
}