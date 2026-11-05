#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#pragma OPENCL FP_CONTRACT OFF

__kernel void kinera_linear_forward_f64(
    __global const double *weights,
    __global const double *bias,
    __global const double *inputs,
    __global double *output,
    const int input_size,
    const int compensated_sum) {
  const size_t row = get_global_id(0);
  double high = 0.0;
  double low = 0.0;
  const size_t row_offset = row * (size_t)input_size;
  for (int column = 0; column < input_size; ++column) {
    const double value = weights[row_offset + (size_t)column] * inputs[column];
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
  output[row] = high + bias[row];
}
