#include <cuda_runtime.h>

#include <cstddef>
#include <climits>

#if defined(_WIN32)
#define KINERA_CUDA_EXPORT __declspec(dllexport)
#else
#define KINERA_CUDA_EXPORT __attribute__((visibility("default")))
#endif

namespace {

constexpr int kThreadsPerBlock = 128;

constexpr std::size_t hybrid_grid_block_count(
    std::size_t output_count) noexcept {
  return output_count / kThreadsPerBlock +
         static_cast<std::size_t>(output_count % kThreadsPerBlock != 0);
}

__global__ void hybrid_representation_forward_f64_kernel(
    const double* char_vectors, const double* feature_vectors,
    const double* syllable_vectors, double* output, int context_size,
    int char_dim, int feature_dim, int syllable_dim, int output_dim) {
  const std::size_t output_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t output_count =
      static_cast<std::size_t>(context_size) * output_dim;
  if (output_index >= output_count) {
    return;
  }

  const int position = static_cast<int>(output_index / output_dim);
  const int column = static_cast<int>(output_index % output_dim);
  if (column < char_dim) {
    const std::size_t input_index =
        static_cast<std::size_t>(position) * char_dim + column;
    output[output_index] = char_vectors[input_index];
    return;
  }

  const int feature_column = column - char_dim;
  if (feature_column < feature_dim) {
    const std::size_t input_index =
        static_cast<std::size_t>(position) * feature_dim + feature_column;
    output[output_index] = feature_vectors[input_index];
    return;
  }

  const int syllable_column = feature_column - feature_dim;
  const std::size_t input_index =
      static_cast<std::size_t>(position) * syllable_dim + syllable_column;
  output[output_index] = syllable_vectors[input_index];
}

}  // namespace

extern "C" KINERA_CUDA_EXPORT int
kinera_cuda_hybrid_representation_forward_f64(
    int device_id, const double* char_vectors, const double* feature_vectors,
    const double* syllable_vectors, double* output, int context_size,
    int char_dim, int feature_dim, int syllable_dim) noexcept {
  if (device_id < 0 || char_vectors == nullptr || feature_vectors == nullptr ||
      syllable_vectors == nullptr || output == nullptr || context_size <= 0 ||
      char_dim <= 0 || feature_dim <= 0 || syllable_dim <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const long long output_dim_wide = static_cast<long long>(char_dim) +
                                    static_cast<long long>(feature_dim) +
                                    static_cast<long long>(syllable_dim);
  if (output_dim_wide > 2147483647LL) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int output_dim = static_cast<int>(output_dim_wide);
  const std::size_t output_count =
      static_cast<std::size_t>(context_size) * output_dim;
  const std::size_t blocks_wide = hybrid_grid_block_count(output_count);
  if (blocks_wide > static_cast<std::size_t>(INT_MAX)) {
    return static_cast<int>(cudaErrorInvalidConfiguration);
  }
  const unsigned int blocks = static_cast<unsigned int>(blocks_wide);

  cudaError_t status = cudaSetDevice(device_id);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  hybrid_representation_forward_f64_kernel<<<blocks, kThreadsPerBlock>>>(
      char_vectors, feature_vectors, syllable_vectors, output, context_size,
      char_dim, feature_dim, syllable_dim, output_dim);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaDeviceSynchronize());
}
