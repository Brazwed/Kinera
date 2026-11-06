#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#pragma OPENCL FP_CONTRACT OFF

__kernel void kinera_embedding_step_f64(
    __global double *weights,
    __global const double *grad_weights,
    const int vocab_size,
    const int embedding_dim,
    const double learning_rate) {
  const size_t element = get_global_id(0);
  const size_t weight_count = (size_t)vocab_size * (size_t)embedding_dim;
  if (element >= weight_count) {
    return;
  }
  const double update = learning_rate * grad_weights[element];
  weights[element] = weights[element] - update;
}
