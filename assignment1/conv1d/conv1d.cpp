#include <torch/extension.h>

#include <vector>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.dtype() == torch::kFloat16, #x "must be float16 tensor")

torch::Tensor conv1d_cuda_half(
    torch::Tensor u,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding);

torch::Tensor conv1d(
    torch::Tensor u,
    torch::Tensor weight,
    torch::Tensor bias,
    uint padding)
{
    CHECK_INPUT(u);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    return conv1d_cuda_half(u, weight, bias, padding);

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &conv1d, "short_filter forward (CUDA)");
}