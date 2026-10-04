#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

#if defined(_WIN32)
#define KINERA_CUDA_EXPORT __declspec(dllexport)
#else
#define KINERA_CUDA_EXPORT __attribute__((visibility("default")))
#endif

namespace {

__global__ void embedding_step_f64_kernel(
    double* weights, const double* grad_weights, double learning_rate,
    std::size_t weight_count) {
  const std::size_t weight_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (weight_index >= weight_count) {
    return;
  }
  weights[weight_index] =
      weights[weight_index] - learning_rate * grad_weights[weight_index];
}

}  // namespace

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_embedding_step_f64(
    int device_id, double* weights, const double* grad_weights,
    double learning_rate, int vocab_size, int embedding_dim) noexcept {
  if (device_id < 0 || weights == nullptr || grad_weights == nullptr ||
      !std::isfinite(learning_rate) || learning_rate < 0.0 || vocab_size <= 0 ||
      embedding_dim <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t status = cudaSetDevice(device_id);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  constexpr int threads_per_block = 128;
  const std::size_t weight_count =
      static_cast<std::size_t>(vocab_size) * embedding_dim;
  const unsigned int blocks = static_cast<unsigned int>(
      (weight_count + threads_per_block - 1) / threads_per_block);
  embedding_step_f64_kernel<<<blocks, threads_per_block>>>(
      weights, grad_weights, learning_rate, weight_count);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaDeviceSynchronize());
}
