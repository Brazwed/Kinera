#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define KINERA_CUDA_EXPORT __declspec(dllexport)
#else
#define KINERA_CUDA_EXPORT __attribute__((visibility("default")))
#endif

namespace {

__global__ void embedding_backward_f64_kernel(
    const int32_t* ids, const double* grad_output, double* grad_weights,
    int vocab_size, int embedding_dim, int ids_count) {
  const std::size_t gradient_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t gradient_count =
      static_cast<std::size_t>(vocab_size) * embedding_dim;
  if (gradient_index >= gradient_count) {
    return;
  }

  const int vocab_row = static_cast<int>(gradient_index / embedding_dim);
  const int dimension = static_cast<int>(gradient_index % embedding_dim);
  double accumulated = grad_weights[gradient_index];
  for (int position = 0; position < ids_count; ++position) {
    if (ids[position] == vocab_row) {
      const std::size_t output_index =
          static_cast<std::size_t>(position) * embedding_dim + dimension;
      accumulated = accumulated + grad_output[output_index];
    }
  }
  grad_weights[gradient_index] = accumulated;
}

}  // namespace

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_embedding_backward_f64(
    int device_id, const int32_t* ids, const double* grad_output,
    double* grad_weights, int vocab_size, int embedding_dim,
    int ids_count) noexcept {
  if (device_id < 0 || ids == nullptr || grad_output == nullptr ||
      grad_weights == nullptr || vocab_size <= 0 || embedding_dim <= 0 ||
      ids_count <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t status = cudaSetDevice(device_id);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  constexpr int threads_per_block = 128;
  const std::size_t gradient_count =
      static_cast<std::size_t>(vocab_size) * embedding_dim;
  const unsigned int blocks = static_cast<unsigned int>(
      (gradient_count + threads_per_block - 1) / threads_per_block);
  embedding_backward_f64_kernel<<<blocks, threads_per_block>>>(
      ids, grad_output, grad_weights, vocab_size, embedding_dim, ids_count);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaDeviceSynchronize());
}
