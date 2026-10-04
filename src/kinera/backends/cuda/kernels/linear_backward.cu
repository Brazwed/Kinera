#include <cuda_runtime.h>

#if defined(_WIN32)
#define KINERA_CUDA_EXPORT __declspec(dllexport)
#else
#define KINERA_CUDA_EXPORT __attribute__((visibility("default")))
#endif

namespace {

__global__ void linear_backward_parameter_kernel(
    const double* inputs, const double* grad_output, double* grad_weights,
    double* grad_bias, int input_size, int output_size) {
  const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= output_size) {
    return;
  }

  const double row_gradient = grad_output[output_index];
  grad_bias[output_index] = row_gradient;
  const int weight_offset = output_index * input_size;
  for (int input_index = 0; input_index < input_size; ++input_index) {
    grad_weights[weight_offset + input_index] =
        row_gradient * inputs[input_index];
  }
}

__global__ void linear_backward_input_kernel(
    const double* weights, const double* grad_output, double* grad_inputs,
    int input_size, int output_size) {
  const int input_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (input_index >= input_size) {
    return;
  }

  double accumulated = 0.0;
  for (int output_index = 0; output_index < output_size; ++output_index) {
    const double product =
        weights[output_index * input_size + input_index] *
        grad_output[output_index];
    accumulated = accumulated + product;
  }
  grad_inputs[input_index] = accumulated;
}

}  // namespace

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_linear_backward_f64(
    int device_id, const double* weights, const double* inputs,
    const double* grad_output, double* grad_weights, double* grad_bias,
    double* grad_inputs, int input_size, int output_size) noexcept {
  if (device_id < 0 || weights == nullptr || inputs == nullptr ||
      grad_output == nullptr || grad_weights == nullptr || grad_bias == nullptr ||
      grad_inputs == nullptr || input_size <= 0 || output_size <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t status = cudaSetDevice(device_id);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  constexpr int threads_per_block = 128;
  const int output_blocks = (output_size - 1) / threads_per_block + 1;
  const int input_blocks = (input_size - 1) / threads_per_block + 1;
  linear_backward_parameter_kernel<<<output_blocks, threads_per_block>>>(
      inputs, grad_output, grad_weights, grad_bias, input_size, output_size);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  linear_backward_input_kernel<<<input_blocks, threads_per_block>>>(
      weights, grad_output, grad_inputs, input_size, output_size);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaDeviceSynchronize());
}
