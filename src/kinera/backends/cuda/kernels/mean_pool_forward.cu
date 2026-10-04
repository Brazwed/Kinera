#include <cuda_runtime.h>

#include <cmath>
#include <climits>
#include <cstddef>

#if defined(_WIN32)
#define KINERA_CUDA_EXPORT __declspec(dllexport)
#else
#define KINERA_CUDA_EXPORT __attribute__((visibility("default")))
#endif

namespace {

constexpr int kThreadsPerBlock = 128;

constexpr unsigned int mean_pool_block_count(int feature_dim) noexcept {
  return static_cast<unsigned int>(
      (static_cast<long long>(feature_dim) + kThreadsPerBlock - 1LL) /
      kThreadsPerBlock);
}

__host__ __device__ constexpr std::size_t mean_pool_feature_index(
    unsigned int block_index, unsigned int thread_index) noexcept {
  return static_cast<std::size_t>(block_index) * kThreadsPerBlock +
         thread_index;
}

static_assert(mean_pool_block_count(INT_MAX) == 16777216U,
              "mean pool grid must cover every positive int dimension");
static_assert(mean_pool_feature_index(mean_pool_block_count(INT_MAX) - 1U,
                                      kThreadsPerBlock - 1U) ==
                  static_cast<std::size_t>(INT_MAX),
              "mean pool lane index must not wrap at INT_MAX");

__global__ void mean_pool_forward_f64_kernel(
    const double* vectors, double* pooled, int context_size,
    int feature_dim, int compensated_sum) {
  const std::size_t feature =
      mean_pool_feature_index(blockIdx.x, threadIdx.x);
  if (feature >= static_cast<std::size_t>(feature_dim)) {
    return;
  }

  // Match the compensated floating-point sum used by the CPU oracle's sum().
  double high = 0.0;
  double low = 0.0;
  for (int position = 0; position < context_size; ++position) {
    const std::size_t input_index =
        static_cast<std::size_t>(position) * feature_dim + feature;
    const double value = vectors[input_index];
    const double next = high + value;
    if (compensated_sum != 0) {
      if (fabs(high) >= fabs(value)) {
        low += (high - next) + value;
      } else {
        low += (value - next) + high;
      }
    }
    high = next;
  }

  double sum = high;
  if (compensated_sum != 0 && low != 0.0 && isfinite(low)) {
    sum += low;
  }
  pooled[feature] = sum / static_cast<double>(context_size);
}

}  // namespace

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_mean_pool_forward_f64(
    int device_id, const double* vectors, double* pooled, int context_size,
    int feature_dim, int compensated_sum) noexcept {
  if (device_id < 0 || vectors == nullptr || pooled == nullptr ||
      context_size <= 0 || feature_dim <= 0 || compensated_sum < 0 ||
      compensated_sum > 1) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t status = cudaSetDevice(device_id);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  const unsigned int blocks = mean_pool_block_count(feature_dim);
  mean_pool_forward_f64_kernel<<<blocks, kThreadsPerBlock>>>(
      vectors, pooled, context_size, feature_dim, compensated_sum);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaDeviceSynchronize());
}
