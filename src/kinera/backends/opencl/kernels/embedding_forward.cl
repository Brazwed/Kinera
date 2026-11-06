#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void kinera_embedding_validate_ids_i32(
    __global const int *ids,
    __global int *status,
    const int ids_count,
    const int vocab_size) {
  if (get_global_id(0) != 0) {
    return;
  }
  int invalid_position = -1;
  for (int position = 0; position < ids_count; ++position) {
    const int token_id = ids[position];
    if (token_id < 0 || token_id >= vocab_size) {
      invalid_position = position;
      break;
    }
  }
  status[0] = invalid_position;
}

__kernel void kinera_embedding_forward_f64(
    __global const double *weights,
    __global const int *ids,
    __global double *output,
    const int embedding_dim,
    const int ids_count) {
  const size_t element = get_global_id(0);
  const size_t total_elements = (size_t)ids_count * (size_t)embedding_dim;
  if (element >= total_elements) {
    return;
  }
  const size_t output_row = element / (size_t)embedding_dim;
  const size_t dimension = element % (size_t)embedding_dim;
  const size_t weight_index = (size_t)ids[output_row] * (size_t)embedding_dim + dimension;
  output[element] = weights[weight_index];
}
