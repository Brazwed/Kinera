#include <cuda_runtime.h>

#include <cmath>

#if defined(_WIN32)
#define KINERA_CUDA_EXPORT __declspec(dllexport)
#else
#define KINERA_CUDA_EXPORT __attribute__((visibility("default")))
#endif

namespace {

constexpr int KINERA_CUDA_INVALID_NUMERIC_STATUS = -2;

__device__ double exp_sum(
    const double* logits, int logits_count, double maximum,
    int compensated_sum) {
  double high = 0.0;
  double low = 0.0;
  for (int index = 0; index < logits_count; ++index) {
    const double value = exp(logits[index] - maximum);
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
  if (compensated_sum != 0 && low != 0.0 && isfinite(low)) {
    high += low;
  }
  return high;
}

__global__ void cross_entropy_forward_f64_kernel(
    const double* logits, double* loss, int* status, int logits_count,
    int target_id, int compensated_sum) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  *status = 0;
  double maximum = logits[0];
  if (!isfinite(maximum)) {
    *status = KINERA_CUDA_INVALID_NUMERIC_STATUS;
    return;
  }
  for (int index = 1; index < logits_count; ++index) {
    const double value = logits[index];
    if (!isfinite(value)) {
      *status = KINERA_CUDA_INVALID_NUMERIC_STATUS;
      return;
    }
    if (value > maximum) {
      maximum = value;
    }
  }
  const double denominator =
      exp_sum(logits, logits_count, maximum, compensated_sum);
  const double result = maximum + log(denominator) - logits[target_id];
  if (!isfinite(result)) {
    *status = KINERA_CUDA_INVALID_NUMERIC_STATUS;
    return;
  }
  *loss = result;
}

__global__ void cross_entropy_gradient_f64_kernel(
    const double* logits, double* gradient, int* status, int logits_count,
    int target_id, double scale, int compensated_sum) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  *status = 0;
  double maximum = logits[0];
  if (!isfinite(maximum)) {
    *status = KINERA_CUDA_INVALID_NUMERIC_STATUS;
    return;
  }
  for (int index = 1; index < logits_count; ++index) {
    const double value = logits[index];
    if (!isfinite(value)) {
      *status = KINERA_CUDA_INVALID_NUMERIC_STATUS;
      return;
    }
    if (value > maximum) {
      maximum = value;
    }
  }
  const double denominator =
      exp_sum(logits, logits_count, maximum, compensated_sum);
  for (int index = 0; index < logits_count; ++index) {
    double value = exp(logits[index] - maximum) / denominator;
    if (index == target_id) {
      value = value - 1.0;
    }
    const double scaled = scale * value;
    if (!isfinite(scaled)) {
      *status = KINERA_CUDA_INVALID_NUMERIC_STATUS;
      return;
    }
    gradient[index] = scaled;
  }
}

__global__ void binary_cross_entropy_with_logits_forward_f64_kernel(
    const double* logit, double* loss, int* status, double target) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  *status = 0;
  const double value = logit[0];
  if (!isfinite(value)) {
    *status = KINERA_CUDA_INVALID_NUMERIC_STATUS;
    return;
  }
  const double result =
      fmax(value, 0.0) - value * target + log1p(exp(-fabs(value)));
  if (!isfinite(result)) {
    *status = KINERA_CUDA_INVALID_NUMERIC_STATUS;
    return;
  }
  *loss = result;
}

__global__ void binary_cross_entropy_with_logits_gradient_f64_kernel(
    const double* logit, double* gradient, int* status, double target,
    double scale) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  *status = 0;
  const double value = logit[0];
  if (!isfinite(value)) {
    *status = KINERA_CUDA_INVALID_NUMERIC_STATUS;
    return;
  }
  double sigmoid = 0.0;
  if (value >= 0.0) {
    const double exponential = exp(-value);
    sigmoid = 1.0 / (1.0 + exponential);
  } else {
    const double exponential = exp(value);
    sigmoid = exponential / (1.0 + exponential);
  }
  const double result = scale * (sigmoid - target);
  if (!isfinite(result)) {
    *status = KINERA_CUDA_INVALID_NUMERIC_STATUS;
    return;
  }
  gradient[0] = result;
}

cudaError_t select_device(int device_id) noexcept {
  if (device_id < 0) {
    return cudaErrorInvalidDevice;
  }
  return cudaSetDevice(device_id);
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

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_cross_entropy_forward_f64(
    int device_id, const double* logits, double* loss, int* status,
    int logits_count, int target_id, int compensated_sum) noexcept {
  if (logits == nullptr || loss == nullptr || status == nullptr ||
      logits_count <= 0 || target_id < 0 || target_id >= logits_count ||
      compensated_sum < 0 || compensated_sum > 1) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaError_t error = select_device(device_id);
  if (error != cudaSuccess) {
    return static_cast<int>(error);
  }
  cross_entropy_forward_f64_kernel<<<1, 1>>>(
      logits, loss, status, logits_count, target_id, compensated_sum);
  return finish_kernel(cudaSuccess);
}

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_cross_entropy_gradient_f64(
    int device_id, const double* logits, double* gradient, int* status,
    int logits_count, int target_id, double scale,
    int compensated_sum) noexcept {
  if (logits == nullptr || gradient == nullptr || status == nullptr ||
      logits_count <= 0 || target_id < 0 || target_id >= logits_count ||
      !isfinite(scale) || scale < 0.0 || compensated_sum < 0 ||
      compensated_sum > 1) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaError_t error = select_device(device_id);
  if (error != cudaSuccess) {
    return static_cast<int>(error);
  }
  cross_entropy_gradient_f64_kernel<<<1, 1>>>(
      logits, gradient, status, logits_count, target_id, scale,
      compensated_sum);
  return finish_kernel(cudaSuccess);
}

extern "C" KINERA_CUDA_EXPORT int
kinera_cuda_binary_cross_entropy_with_logits_forward_f64(
    int device_id, const double* logit, double* loss, int* status,
    double target) noexcept {
  if (logit == nullptr || loss == nullptr || status == nullptr ||
      !isfinite(target) || target < 0.0 || target > 1.0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaError_t error = select_device(device_id);
  if (error != cudaSuccess) {
    return static_cast<int>(error);
  }
  binary_cross_entropy_with_logits_forward_f64_kernel<<<1, 1>>>(
      logit, loss, status, target);
  return finish_kernel(cudaSuccess);
}

extern "C" KINERA_CUDA_EXPORT int
kinera_cuda_binary_cross_entropy_with_logits_gradient_f64(
    int device_id, const double* logit, double* gradient, int* status,
    double target, double scale) noexcept {
  if (logit == nullptr || gradient == nullptr || status == nullptr ||
      !isfinite(target) || target < 0.0 || target > 1.0 || !isfinite(scale) ||
      scale < 0.0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaError_t error = select_device(device_id);
  if (error != cudaSuccess) {
    return static_cast<int>(error);
  }
  binary_cross_entropy_with_logits_gradient_f64_kernel<<<1, 1>>>(
      logit, gradient, status, target, scale);
  return finish_kernel(cudaSuccess);
}
