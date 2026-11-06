#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kinera_hybrid_representation_validate_finite_f64(
    __global const double* char_vectors,
    __global const double* feature_vectors,
    __global const double* syllable_vectors,
    __global int* status,
    const int context_size,
    const int char_dim,
    const int feature_dim,
    const int syllable_dim
) {
    if (get_global_id(0) != 0) {
        return;
    }

    status[0] = 0;
    const int dimensions[3] = {char_dim, feature_dim, syllable_dim};

    for (int channel = 0; channel < 3; ++channel) {
        const int value_count = context_size * dimensions[channel];
        for (int index = 0; index < value_count; ++index) {
            double value;
            if (channel == 0) {
                value = char_vectors[index];
            } else if (channel == 1) {
                value = feature_vectors[index];
            } else {
                value = syllable_vectors[index];
            }
            if (!isfinite(value)) {
                status[0] = channel + 1;
                return;
            }
        }
    }
}

__kernel void kinera_hybrid_representation_forward_f64(
    __global const double* char_vectors,
    __global const double* feature_vectors,
    __global const double* syllable_vectors,
    __global double* output,
    const int context_size,
    const int char_dim,
    const int feature_dim,
    const int syllable_dim
) {
    const size_t index = get_global_id(0);
    const int output_dim = char_dim + feature_dim + syllable_dim;
    const size_t output_count = (size_t)context_size * (size_t)output_dim;
    if (index >= output_count) {
        return;
    }

    const int row = (int)(index / (size_t)output_dim);
    const int column = (int)(index % (size_t)output_dim);
    if (column < char_dim) {
        output[index] = char_vectors[(size_t)row * (size_t)char_dim + column];
    } else if (column < char_dim + feature_dim) {
        const int feature_column = column - char_dim;
        output[index] = feature_vectors[
            (size_t)row * (size_t)feature_dim + feature_column
        ];
    } else {
        const int syllable_column = column - char_dim - feature_dim;
        output[index] = syllable_vectors[
            (size_t)row * (size_t)syllable_dim + syllable_column
        ];
    }
}
