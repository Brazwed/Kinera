#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#pragma OPENCL FP_CONTRACT OFF

__kernel void kinera_linear_backward_f64(
    __global const double *weights,
    __global const double *inputs,
    __global const double *grad_output,
    __global double *grad_weights,
    __global double *grad_bias,
    __global double *grad_inputs,
    const int output_size,
    const int input_size,
    const int compensated_sum) {
  const size_t index = get_global_id(0);
  const size_t weights_count = (size_t)output_size * (size_t)input_size;

  if (index < weights_count) {
    const size_t row = index / (size_t)input_size;
    const size_t column = index % (size_t)input_size;
    grad_weights[index] = grad_output[row] * inputs[column];
  }

  if (index < (size_t)output_size) {
    grad_bias[index] = grad_output[index];
  }

  if (index < (size_t)input_size) {
    double high = 0.0;
    double low = 0.0;
    for (int row = 0; row < output_size; ++row) {
      const size_t weight_index = (size_t)row * (size_t)input_size + index;
      const double value = weights[weight_index] * grad_output[row];
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
    grad_inputs[index] = high;
  }
}
