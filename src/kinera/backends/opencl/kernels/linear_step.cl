#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#pragma OPENCL FP_CONTRACT OFF

__kernel void kinera_linear_step_f64(
    __global double *weights,
    __global double *bias,
    __global const double *grad_weights,
    __global const double *grad_bias,
    const int output_size,
    const int input_size,
    const double learning_rate) {
  const size_t index = get_global_id(0);
  const size_t weights_count = (size_t)output_size * (size_t)input_size;

  if (index < weights_count) {
    const double update = learning_rate * grad_weights[index];
    weights[index] = weights[index] - update;
  }
  if (index < (size_t)output_size) {
    const double update = learning_rate * grad_bias[index];
    bias[index] = bias[index] - update;
  }
}
