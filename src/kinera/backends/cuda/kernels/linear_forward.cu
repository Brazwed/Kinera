#include <cuda_runtime.h>

#if defined(_WIN32)
#define KINERA_CUDA_EXPORT __declspec(dllexport)
#else
#define KINERA_CUDA_EXPORT __attribute__((visibility("default")))
#endif

namespace {

__global__ void linear_forward_f64_kernel(
    const double* weights, const double* bias, const double* inputs,
    double* outputs, int input_size, int output_size) {
  const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= output_size) {
    return;
  }

  double accumulated = 0.0;
  const int weight_offset = output_index * input_size;
  for (int input_index = 0; input_index < input_size; ++input_index) {
    const double product =
        weights[weight_offset + input_index] * inputs[input_index];
    accumulated = accumulated + product;
  }
  outputs[output_index] = accumulated + bias[output_index];
}

}  // namespace

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_linear_forward_f64(
    int device_id, const double* weights, const double* bias,
    const double* inputs, double* outputs, int input_size,
    int output_size) noexcept {
  if (device_id < 0 || weights == nullptr || bias == nullptr ||
      inputs == nullptr || outputs == nullptr || input_size <= 0 ||
      output_size <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t status = cudaSetDevice(device_id);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  constexpr int threads_per_block = 128;
  const int blocks = (output_size + threads_per_block - 1) / threads_per_block;
  linear_forward_f64_kernel<<<blocks, threads_per_block>>>(
      weights, bias, inputs, outputs, input_size, output_size);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaDeviceSynchronize());
}
