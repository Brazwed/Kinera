#include <cuda_runtime.h>

#include <climits>
#include <cstddef>

#if defined(_WIN32)
#define KINERA_CUDA_EXPORT __declspec(dllexport)
#else
#define KINERA_CUDA_EXPORT __attribute__((visibility("default")))
#endif

namespace {

constexpr int kThreadsPerBlock = 128;

constexpr std::size_t grid_block_count(std::size_t element_count) noexcept {
  return element_count / kThreadsPerBlock +
         static_cast<std::size_t>(element_count % kThreadsPerBlock != 0);
}

static_assert(grid_block_count(INT_MAX) == 16777216U,
              "resident backward grids must cover every positive int dimension");

__host__ __device__ constexpr std::size_t global_index(
    unsigned int block_index, unsigned int thread_index) noexcept {
  return static_cast<std::size_t>(block_index) * kThreadsPerBlock +
         thread_index;
}

__device__ void compensated_add(
    double value, double& high, double& low, int compensated_sum) {
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

__global__ void sum_head_input_gradients_f64_kernel(
    const double* char_gradient, const double* syllable_gradient,
    const double* boundary_gradient, double* output, int feature_dim,
    int compensated_sum) {
  const std::size_t feature = global_index(blockIdx.x, threadIdx.x);
  if (feature >= static_cast<std::size_t>(feature_dim)) {
    return;
  }
  double high = 0.0;
  double low = 0.0;
  compensated_add(char_gradient[feature], high, low, compensated_sum);
  compensated_add(syllable_gradient[feature], high, low, compensated_sum);
  compensated_add(boundary_gradient[feature], high, low, compensated_sum);
  if (compensated_sum != 0 && low != 0.0 && isfinite(low)) {
    high += low;
  }
  output[feature] = high;
}

__global__ void mean_pool_backward_f64_kernel(
    const double* grad_pooled, double* grad_vectors, int context_size,
    int feature_dim) {
  const std::size_t element = global_index(blockIdx.x, threadIdx.x);
  const std::size_t element_count =
      static_cast<std::size_t>(context_size) * feature_dim;
  if (element >= element_count) {
    return;
  }
  const std::size_t feature = element % static_cast<std::size_t>(feature_dim);
  grad_vectors[element] = grad_pooled[feature] / static_cast<double>(context_size);
}

__global__ void hybrid_representation_backward_f64_kernel(
    const double* grad_output, double* char_gradient, double* feature_gradient,
    double* syllable_gradient, int context_size, int char_dim, int feature_dim,
    int syllable_dim, int output_dim) {
  const std::size_t element = global_index(blockIdx.x, threadIdx.x);
  const std::size_t element_count =
      static_cast<std::size_t>(context_size) * output_dim;
  if (element >= element_count) {
    return;
  }
  const std::size_t row = element / static_cast<std::size_t>(output_dim);
  const int channel = static_cast<int>(element % output_dim);
  const double value = grad_output[element];
  if (channel < char_dim) {
    const std::size_t destination =
        row * static_cast<std::size_t>(char_dim) + channel;
    char_gradient[destination] = value;
  } else if (channel < char_dim + feature_dim) {
    const int feature = channel - char_dim;
    const std::size_t destination =
        row * static_cast<std::size_t>(feature_dim) + feature;
    feature_gradient[destination] = value;
  } else {
    const int feature = channel - char_dim - feature_dim;
    const std::size_t destination =
        row * static_cast<std::size_t>(syllable_dim) + feature;
    syllable_gradient[destination] = value;
  }
}

cudaError_t select_device(int device_id) noexcept {
  if (device_id < 0) {
    return cudaErrorInvalidDevice;
  }
  return cudaSetDevice(device_id);
}

int launch_blocks(std::size_t element_count, unsigned int* blocks) noexcept {
  if (blocks == nullptr || element_count == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t blocks_wide = grid_block_count(element_count);
  if (blocks_wide > static_cast<std::size_t>(INT_MAX)) {
    return static_cast<int>(cudaErrorInvalidConfiguration);
  }
  *blocks = static_cast<unsigned int>(blocks_wide);
  return static_cast<int>(cudaSuccess);
}

int finish_kernel(cudaError_t status) noexcept {
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaDeviceSynchronize());
}

}  // namespace

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_sum_head_input_gradients_f64(
    int device_id, const double* char_gradient,
    const double* syllable_gradient, const double* boundary_gradient,
    double* output, int feature_dim, int compensated_sum) noexcept {
  if (char_gradient == nullptr || syllable_gradient == nullptr ||
      boundary_gradient == nullptr || output == nullptr || feature_dim <= 0 ||
      compensated_sum < 0 || compensated_sum > 1) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  unsigned int blocks = 0;
  int status = launch_blocks(static_cast<std::size_t>(feature_dim), &blocks);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  cudaError_t error = select_device(device_id);
  if (error != cudaSuccess) {
    return static_cast<int>(error);
  }
  sum_head_input_gradients_f64_kernel<<<blocks, kThreadsPerBlock>>>(
      char_gradient, syllable_gradient, boundary_gradient, output, feature_dim,
      compensated_sum);
  return finish_kernel(cudaSuccess);
}

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_mean_pool_backward_f64(
    int device_id, const double* grad_pooled, double* grad_vectors,
    int context_size, int feature_dim) noexcept {
  if (grad_pooled == nullptr || grad_vectors == nullptr || context_size <= 0 ||
      feature_dim <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t element_count =
      static_cast<std::size_t>(context_size) * feature_dim;
  unsigned int blocks = 0;
  int status = launch_blocks(element_count, &blocks);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  cudaError_t error = select_device(device_id);
  if (error != cudaSuccess) {
    return static_cast<int>(error);
  }
  mean_pool_backward_f64_kernel<<<blocks, kThreadsPerBlock>>>(
      grad_pooled, grad_vectors, context_size, feature_dim);
  return finish_kernel(cudaSuccess);
}

extern "C" KINERA_CUDA_EXPORT int
kinera_cuda_hybrid_representation_backward_f64(
    int device_id, const double* grad_output, double* char_gradient,
    double* feature_gradient, double* syllable_gradient, int context_size,
    int char_dim, int feature_dim, int syllable_dim) noexcept {
  if (grad_output == nullptr || char_gradient == nullptr ||
      feature_gradient == nullptr || syllable_gradient == nullptr ||
      context_size <= 0 || char_dim <= 0 || feature_dim <= 0 ||
      syllable_dim <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const long long output_dim_wide = static_cast<long long>(char_dim) +
                                    feature_dim + syllable_dim;
  if (output_dim_wide > INT_MAX) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int output_dim = static_cast<int>(output_dim_wide);
  const std::size_t element_count =
      static_cast<std::size_t>(context_size) * output_dim;
  unsigned int blocks = 0;
  int status = launch_blocks(element_count, &blocks);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  cudaError_t error = select_device(device_id);
  if (error != cudaSuccess) {
    return static_cast<int>(error);
  }
  hybrid_representation_backward_f64_kernel<<<blocks, kThreadsPerBlock>>>(
      grad_output, char_gradient, feature_gradient, syllable_gradient,
      context_size, char_dim, feature_dim, syllable_dim, output_dim);
  return finish_kernel(cudaSuccess);
}
