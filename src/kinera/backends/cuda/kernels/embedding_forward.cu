#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define KINERA_CUDA_EXPORT __declspec(dllexport)
#else
#define KINERA_CUDA_EXPORT __attribute__((visibility("default")))
#endif

namespace {

__global__ void embedding_forward_f64_kernel(
    const double* weights, const int32_t* ids, double* output,
    int embedding_dim, int ids_count) {
  const std::size_t output_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t output_count =
      static_cast<std::size_t>(ids_count) * embedding_dim;
  if (output_index >= output_count) {
    return;
  }

  const int position = static_cast<int>(output_index / embedding_dim);
  const int column = static_cast<int>(output_index % embedding_dim);
  const int row = ids[position];
  const std::size_t weight_index =
      static_cast<std::size_t>(row) * embedding_dim + column;
  output[output_index] = weights[weight_index];
}

}  // namespace

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_embedding_forward_f64(
    int device_id, const double* weights, const int32_t* ids, double* output,
    int vocab_size, int embedding_dim, int ids_count) noexcept {
  if (device_id < 0 || weights == nullptr || ids == nullptr ||
      output == nullptr || vocab_size <= 0 || embedding_dim <= 0 ||
      ids_count <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t status = cudaSetDevice(device_id);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  constexpr int threads_per_block = 128;
  const std::size_t output_count =
      static_cast<std::size_t>(ids_count) * embedding_dim;
  const unsigned int blocks = static_cast<unsigned int>(
      (output_count + threads_per_block - 1) / threads_per_block);
  embedding_forward_f64_kernel<<<blocks, threads_per_block>>>(
      weights, ids, output, embedding_dim, ids_count);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaDeviceSynchronize());
}
