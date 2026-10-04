#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>

#if defined(_WIN32)
#define KINERA_CUDA_EXPORT __declspec(dllexport)
#else
#define KINERA_CUDA_EXPORT __attribute__((visibility("default")))
#endif

namespace {

cudaError_t select_device(int device_id) noexcept {
  if (device_id < 0) {
    return cudaErrorInvalidDevice;
  }
  return cudaSetDevice(device_id);
}

}  // namespace

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_abi_version() noexcept {
  return 3;
}

extern "C" KINERA_CUDA_EXPORT const char* kinera_cuda_error_string(
    int status) noexcept {
  return cudaGetErrorString(static_cast<cudaError_t>(status));
}

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_get_device_count(
    int* count) noexcept {
  if (count == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *count = 0;
  return static_cast<int>(cudaGetDeviceCount(count));
}

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_get_device_info(
    int device_id, char* name, std::size_t name_capacity, int* major,
    int* minor, unsigned long long* total_memory_bytes) noexcept {
  if (name == nullptr || name_capacity == 0 || major == nullptr ||
      minor == nullptr || total_memory_bytes == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaDeviceProp properties{};
  cudaError_t status = cudaGetDeviceProperties(&properties, device_id);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  std::strncpy(name, properties.name, name_capacity - 1);
  name[name_capacity - 1] = '\0';
  *major = properties.major;
  *minor = properties.minor;
  *total_memory_bytes = static_cast<unsigned long long>(properties.totalGlobalMem);
  return static_cast<int>(cudaSuccess);
}

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_set_device(
    int device_id) noexcept {
  return static_cast<int>(select_device(device_id));
}

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_synchronize(
    int device_id) noexcept {
  cudaError_t status = select_device(device_id);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaDeviceSynchronize());
}

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_allocate(
    int device_id, std::size_t nbytes, void** output) noexcept {
  if (output == nullptr || nbytes == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *output = nullptr;
  cudaError_t status = select_device(device_id);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaMalloc(output, nbytes));
}

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_free(
    int device_id, void* pointer) noexcept {
  if (pointer == nullptr) {
    return static_cast<int>(cudaSuccess);
  }
  cudaError_t status = select_device(device_id);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaFree(pointer));
}

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_copy_to_device(
    int device_id, void* destination, const void* source,
    std::size_t nbytes) noexcept {
  if (destination == nullptr || source == nullptr || nbytes == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaError_t status = select_device(device_id);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaMemcpy(destination, source, nbytes,
                                     cudaMemcpyHostToDevice));
}

extern "C" KINERA_CUDA_EXPORT int kinera_cuda_copy_to_host(
    int device_id, void* destination, const void* source,
    std::size_t nbytes) noexcept {
  if (destination == nullptr || source == nullptr || nbytes == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaError_t status = select_device(device_id);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaMemcpy(destination, source, nbytes,
                                     cudaMemcpyDeviceToHost));
}
