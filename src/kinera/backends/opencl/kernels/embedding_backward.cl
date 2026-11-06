#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

// status == 0 means valid. Positive values encode an invalid ID position + 1;
// negative values encode a non-finite gradient flat index + 1.
__kernel void kinera_embedding_validate_backward_inputs(
    __global const int *ids,
    __global const double *grad_output,
    __global int *status,
    const int ids_count,
    const int vocab_size,
    const int embedding_dim) {
  if (get_global_id(0) != 0) {
    return;
  }
  status[0] = 0;
  for (int position = 0; position < ids_count; ++position) {
    const int token_id = ids[position];
    if (token_id < 0 || token_id >= vocab_size) {
      status[0] = position + 1;
      return;
    }
    for (int dimension = 0; dimension < embedding_dim; ++dimension) {
      const size_t index =
          (size_t)position * (size_t)embedding_dim + (size_t)dimension;
      if (!isfinite(grad_output[index])) {
        status[0] = -((int)index + 1);
        return;
      }
    }
  }
}

// One work-item owns one destination element. It scans IDs in position order,
// matching the CPUBackend's sequential accumulation without floating atomics.
__kernel void kinera_embedding_backward_f64(
    __global double *grad_weights,
    __global const int *ids,
    __global const double *grad_output,
    const int vocab_size,
    const int embedding_dim,
    const int ids_count) {
  const size_t element = get_global_id(0);
  const size_t total_elements = (size_t)vocab_size * (size_t)embedding_dim;
  if (element >= total_elements) {
    return;
  }
  const int row = (int)(element / (size_t)embedding_dim);
  const int dimension = (int)(element % (size_t)embedding_dim);
  double accumulated = grad_weights[element];
  for (int position = 0; position < ids_count; ++position) {
    if (ids[position] == row) {
      const size_t gradient_index =
          (size_t)position * (size_t)embedding_dim + (size_t)dimension;
      accumulated = accumulated + grad_output[gradient_index];
    }
  }
  grad_weights[element] = accumulated;
}
