#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

__kernel void kinera_mean_pool_validate_finite_f64(
    __global const double* vectors,
    __global int* status,
    const int context_size,
    const int feature_dim
) {
    if (get_global_id(0) != 0) {
        return;
    }

    status[0] = 0;
    const size_t value_count =
        (size_t)context_size * (size_t)feature_dim;
    for (size_t index = 0; index < value_count; ++index) {
        if (!isfinite(vectors[index])) {
            status[0] = 1;
            return;
        }
    }
}

__kernel void kinera_mean_pool_forward_f64(
    __global const double* vectors,
    __global double* pooled,
    const int context_size,
    const int feature_dim,
    const int compensated_sum,
    const double inverse_context_size
) {
    const size_t feature = get_global_id(0);
    if (feature >= (size_t)feature_dim) {
        return;
    }

    double high = 0.0;
    double low = 0.0;
    for (int position = 0; position < context_size; ++position) {
        const size_t input_index =
            (size_t)position * (size_t)feature_dim + feature;
        const double value = vectors[input_index];
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

    double sum = high;
    if (compensated_sum != 0 && low != 0.0 && isfinite(low)) {
        sum += low;
    }
    pooled[feature] = sum * inverse_context_size;
}
