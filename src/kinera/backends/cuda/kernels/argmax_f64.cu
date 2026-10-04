#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#if defined(_WIN32)
#define KINERA_CUDA_EXPORT __declspec(dllexport)
#else
#define KINERA_CUDA_EXPORT __attribute__((visibility("default")))
#endif

namespace {

__global__ void argmax_f64_kernel(
    const double* logits, std::int32_t* output_id, int logits_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  double best_value = logits[0];
  if (!isfinite(best_value)) {
    output_id[0] = -1;
    return;
  }
  int best_index = 0;
  for (int index = 1; index < logits_count; ++index) {
    const double value = logits[index];
    if (!isfinite(value)) {
      output_id[0] = -index - 1;
      return;
    }
    // Strict comparison preserves CPU max(..., key=...) first-index ties.
    if (value > best_value) {
      best_value = value;
      best_index = index;
    }
  }
  output_id[0] = best_index;
}

}  // namespace

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_argmax_f64(
    int device_id, const double* logits, std::int32_t* output_id,
    int logits_count) noexcept {
  if (device_id < 0 || logits == nullptr || output_id == nullptr ||
      logits_count <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t status = cudaSetDevice(device_id);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  argmax_f64_kernel<<<1, 1>>>(logits, output_id, logits_count);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaDeviceSynchronize());
}
