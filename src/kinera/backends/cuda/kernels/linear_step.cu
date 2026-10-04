#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

#if defined(_WIN32)
#define KINERA_CUDA_EXPORT __declspec(dllexport)
#else
#define KINERA_CUDA_EXPORT __attribute__((visibility("default")))
#endif

namespace {

__global__ void linear_step_kernel(
    double* weights, double* bias, const double* grad_weights,
    const double* grad_bias, double learning_rate, std::size_t weight_count,
    std::size_t parameter_count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= parameter_count) {
    return;
  }

  if (index < weight_count) {
    weights[index] = weights[index] - learning_rate * grad_weights[index];
    return;
  }

  const std::size_t bias_index = index - weight_count;
  bias[bias_index] = bias[bias_index] - learning_rate * grad_bias[bias_index];
}

}  // namespace

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_linear_step_f64(
    int device_id, double* weights, double* bias, const double* grad_weights,
    const double* grad_bias, double learning_rate, int input_size,
    int output_size) noexcept {
  if (device_id < 0 || weights == nullptr || bias == nullptr ||
      grad_weights == nullptr || grad_bias == nullptr || input_size <= 0 ||
      output_size <= 0 || !std::isfinite(learning_rate) || learning_rate < 0.0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t status = cudaSetDevice(device_id);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (learning_rate == 0.0) {
    return static_cast<int>(cudaSuccess);
  }

  const std::size_t weight_count =
      static_cast<std::size_t>(input_size) * static_cast<std::size_t>(output_size);
  const std::size_t parameter_count =
      weight_count + static_cast<std::size_t>(output_size);
  constexpr unsigned int threads_per_block = 128;
  const unsigned int blocks = static_cast<unsigned int>(
      (parameter_count - 1) / threads_per_block + 1);
  linear_step_kernel<<<blocks, threads_per_block>>>(
      weights, bias, grad_weights, grad_bias, learning_rate, weight_count,
      parameter_count);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaDeviceSynchronize());
}
