#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AMD_VENDOR_ID 0x1002u
#define CL_DEVICE_PCI_BUS_INFO_KHR 0x410Fu
#define SAMPLE_COUNT 16

typedef struct {
    cl_uint domain;
    cl_uint bus;
    cl_uint device;
    cl_uint function;
} KineraPciBusInfo;

typedef struct {
    int supplied;
    unsigned int domain;
    unsigned int bus;
    unsigned int device;
    unsigned int function;
} ExpectedPciAddress;

static const char *kernel_source =
    "#if defined(cl_khr_fp64)\n"
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "#elif defined(cl_amd_fp64)\n"
    "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
    "#endif\n"
    "__kernel void kinera_probe(__global const double *a,\n"
    "                           __global const double *b,\n"
    "                           __global const int *ids,\n"
    "                           __global double *sums,\n"
    "                           __global int *next_ids) {\n"
    "    size_t i = get_global_id(0);\n"
    "    sums[i] = a[i] + b[i];\n"
    "    next_ids[i] = ids[i] + 1;\n"
    "}\n";

static cl_int device_string(cl_device_id device, cl_device_info key,
                            char *value, size_t capacity) {
    size_t actual_size = 0;
    cl_int status = clGetDeviceInfo(device, key, capacity, value, &actual_size);
    if (status != CL_SUCCESS) {
        value[0] = '\0';
        return status;
    }
    if (actual_size > 0 && actual_size <= capacity) {
        value[actual_size - 1] = '\0';
    } else {
        value[capacity - 1] = '\0';
    }
    return CL_SUCCESS;
}

static int has_extension(const char *extensions, const char *wanted) {
    const size_t wanted_length = strlen(wanted);
    const char *current = extensions;

    while ((current = strstr(current, wanted)) != NULL) {
        const int starts_token = current == extensions || current[-1] == ' ';
        const int ends_token = current[wanted_length] == '\0' ||
                               current[wanted_length] == ' ';
        if (starts_token && ends_token) {
            return 1;
        }
        current += wanted_length;
    }
    return 0;
}

static int parse_expected_pci_address(int argc, char **argv,
                                      ExpectedPciAddress *expected) {
    if (argc == 1) {
        return 1;
    }
    if (argc != 3 || strcmp(argv[1], "--pci-bdf") != 0 ||
        sscanf(argv[2], "%x:%x:%x.%x", &expected->domain, &expected->bus,
               &expected->device, &expected->function) != 4 ||
        expected->domain > 0xffffu || expected->bus > 0xffu ||
        expected->device > 0x1fu || expected->function > 7u) {
        fprintf(stderr,
                "Usage: %s [--pci-bdf DOMAIN:BUS:DEVICE.FUNCTION]\n"
                "Example: %s --pci-bdf 0000:03:00.0\n",
                argv[0], argv[0]);
        return 0;
    }
    expected->supplied = 1;
    return 1;
}

static int pci_address_matches(cl_device_id device,
                               const ExpectedPciAddress *expected,
                               KineraPciBusInfo *actual,
                               int *extension_available) {
    KineraPciBusInfo pci_info;
    char extensions[8192];

    if (device_string(device, CL_DEVICE_EXTENSIONS, extensions,
                      sizeof(extensions)) != CL_SUCCESS ||
        !has_extension(extensions, "cl_khr_pci_bus_info")) {
        *extension_available = 0;
        return 0;
    }
    *extension_available = 1;

    if (clGetDeviceInfo(device, CL_DEVICE_PCI_BUS_INFO_KHR,
                        sizeof(pci_info), &pci_info, NULL) != CL_SUCCESS) {
        *extension_available = 0;
        return 0;
    }
    actual->domain = pci_info.domain;
    actual->bus = pci_info.bus;
    actual->device = pci_info.device;
    actual->function = pci_info.function;

    return !expected->supplied ||
           (expected->domain == actual->domain &&
            expected->bus == actual->bus &&
            expected->device == actual->device &&
            expected->function == actual->function);
}

static int is_rx550_name(const char *name) {
    char lower[1024];
    size_t i;

    for (i = 0; i + 1 < sizeof(lower) && name[i] != '\0'; ++i) {
        const char ch = name[i];
        lower[i] = (ch >= 'A' && ch <= 'Z') ? (char)(ch - 'A' + 'a') : ch;
    }
    lower[i] = '\0';
    return strstr(lower, "rx 550") != NULL || strstr(lower, "rx550") != NULL;
}

static void print_opencl_error(const char *operation, cl_int status) {
    fprintf(stderr, "ERROR %s: OpenCL status %d\n", operation, status);
}

int main(int argc, char **argv) {
    ExpectedPciAddress expected = {0};
    cl_platform_id platforms[32];
    cl_uint platform_count = 0;
    cl_device_id selected_device = NULL;
    KineraPciBusInfo selected_pci = {0};
    int selected_has_pci = 0;
    unsigned int matching_gpu_count = 0;
    cl_int status;
    char selected_name[1024] = "";
    char selected_vendor[1024] = "";
    char selected_version[1024] = "";
    char selected_driver[1024] = "";
    char selected_extensions[8192] = "";
    cl_uint selected_vendor_id = 0;
    cl_device_type selected_type = 0;
    cl_device_fp_config double_fp_config = 0;
    cl_ulong global_memory = 0;
    cl_ulong max_allocation = 0;
    cl_ulong local_memory = 0;
    size_t max_work_group_size = 0;
    cl_uint work_item_dimensions = 0;
    size_t work_item_sizes[16] = {0};
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem a_buffer = NULL;
    cl_mem b_buffer = NULL;
    cl_mem ids_buffer = NULL;
    cl_mem sums_buffer = NULL;
    cl_mem next_ids_buffer = NULL;
    double a[SAMPLE_COUNT];
    double b[SAMPLE_COUNT];
    double sums[SAMPLE_COUNT] = {0};
    int32_t ids[SAMPLE_COUNT];
    int32_t next_ids[SAMPLE_COUNT] = {0};
    size_t global_size = SAMPLE_COUNT;
    size_t i;
    int exit_code = 1;

    if (!parse_expected_pci_address(argc, argv, &expected)) {
        return 2;
    }

    status = clGetPlatformIDs(32, platforms, &platform_count);
    if (status != CL_SUCCESS || platform_count == 0) {
        fprintf(stderr,
                "RESULT=NO_GO\n"
                "Reason: no OpenCL platform is registered with the ICD loader.\n"
                "Check the OpenCL loader, ICD vendor file and runtime setup.\n");
        print_opencl_error("clGetPlatformIDs", status);
        return 1;
    }
    if (platform_count > 32) {
        platform_count = 32;
    }
    printf("opencl_platform_count=%u\n", platform_count);

    for (cl_uint p = 0; p < platform_count; ++p) {
        cl_uint gpu_count = 0;
        cl_device_id devices[32];
        char platform_name[1024] = "";

        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME,
                          sizeof(platform_name), platform_name, NULL);
        platform_name[sizeof(platform_name) - 1] = '\0';
        status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 32,
                                devices, &gpu_count);
        if (status == CL_DEVICE_NOT_FOUND) {
            printf("platform[%u]=%s gpu_count=0\n", p, platform_name);
            continue;
        }
        if (status != CL_SUCCESS) {
            print_opencl_error("clGetDeviceIDs(GPU)", status);
            goto cleanup;
        }
        if (gpu_count > 32) {
            gpu_count = 32;
        }
        printf("platform[%u]=%s gpu_count=%u\n", p, platform_name,
               gpu_count);

        for (cl_uint d = 0; d < gpu_count; ++d) {
            char name[1024] = "";
            char vendor[1024] = "";
            cl_uint vendor_id = 0;
            KineraPciBusInfo pci = {0};
            int pci_extension_available = 0;
            const int pci_matches =
                pci_address_matches(devices[d], &expected, &pci,
                                    &pci_extension_available);
            const int named_rx550 =
                device_string(devices[d], CL_DEVICE_NAME, name,
                              sizeof(name)) == CL_SUCCESS &&
                is_rx550_name(name);
            const int identity_matches =
                expected.supplied && pci_extension_available ? pci_matches
                                                              : named_rx550;

            device_string(devices[d], CL_DEVICE_VENDOR, vendor,
                          sizeof(vendor));
            clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR_ID,
                            sizeof(vendor_id), &vendor_id, NULL);

            printf("gpu_candidate[%u].name=%s\n", d, name);
            printf("gpu_candidate[%u].vendor=%s\n", d, vendor);
            printf("gpu_candidate[%u].vendor_id=0x%04x\n", d, vendor_id);
            if (pci_extension_available) {
                printf("gpu_candidate[%u].pci_bdf=%04x:%02x:%02x.%u\n", d,
                       pci.domain, pci.bus, pci.device, pci.function);
                if (expected.supplied) {
                    printf("gpu_candidate[%u].pci_bdf_match=%s\n", d,
                           pci_matches ? "true" : "false");
                }
            } else if (expected.supplied) {
                printf("gpu_candidate[%u].pci_bdf_match=false\n", d);
            }

            if (vendor_id == AMD_VENDOR_ID &&
                identity_matches) {
                ++matching_gpu_count;
                selected_device = devices[d];
                selected_pci = pci;
                selected_has_pci = pci_extension_available;
            }
        }
    }

    if (matching_gpu_count != 1 || selected_device == NULL) {
        fprintf(stderr,
                "RESULT=NO_GO\n"
                "Reason: expected exactly one AMD RX 550 GPU selected by its "
                "device name or PCI bus identity; found %u. CPU devices are "
                "never selected.\n",
                matching_gpu_count);
        goto cleanup;
    }

    device_string(selected_device, CL_DEVICE_NAME, selected_name,
                  sizeof(selected_name));
    device_string(selected_device, CL_DEVICE_VENDOR, selected_vendor,
                  sizeof(selected_vendor));
    device_string(selected_device, CL_DEVICE_VERSION, selected_version,
                  sizeof(selected_version));
    device_string(selected_device, CL_DRIVER_VERSION, selected_driver,
                  sizeof(selected_driver));
    device_string(selected_device, CL_DEVICE_EXTENSIONS, selected_extensions,
                  sizeof(selected_extensions));
    clGetDeviceInfo(selected_device, CL_DEVICE_VENDOR_ID,
                    sizeof(selected_vendor_id), &selected_vendor_id, NULL);
    clGetDeviceInfo(selected_device, CL_DEVICE_TYPE, sizeof(selected_type),
                    &selected_type, NULL);
    clGetDeviceInfo(selected_device, CL_DEVICE_DOUBLE_FP_CONFIG,
                    sizeof(double_fp_config), &double_fp_config, NULL);
    clGetDeviceInfo(selected_device, CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(global_memory), &global_memory, NULL);
    clGetDeviceInfo(selected_device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                    sizeof(max_allocation), &max_allocation, NULL);
    clGetDeviceInfo(selected_device, CL_DEVICE_LOCAL_MEM_SIZE,
                    sizeof(local_memory), &local_memory, NULL);
    clGetDeviceInfo(selected_device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(max_work_group_size), &max_work_group_size, NULL);
    clGetDeviceInfo(selected_device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                    sizeof(work_item_dimensions), &work_item_dimensions, NULL);
    if (work_item_dimensions > 16) {
        fprintf(stderr, "ERROR device reports too many work item dimensions: "
                        "%u\n", work_item_dimensions);
        goto cleanup;
    }
    if (work_item_dimensions > 0) {
        clGetDeviceInfo(selected_device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                        sizeof(size_t) * work_item_dimensions,
                        work_item_sizes, NULL);
    }

    if (!(selected_type & CL_DEVICE_TYPE_GPU) ||
        selected_vendor_id != AMD_VENDOR_ID) {
        fprintf(stderr,
                "RESULT=NO_GO\n"
                "Reason: selected device is not an AMD GPU.\n");
        goto cleanup;
    }

    printf("device_name=%s\n", selected_name);
    printf("device_vendor=%s\n", selected_vendor);
    printf("device_vendor_id=0x%04x\n", selected_vendor_id);
    printf("device_version=%s\n", selected_version);
    printf("driver_version=%s\n", selected_driver);
    printf("device_type=GPU\n");
    if (selected_has_pci) {
        printf("device_pci_bdf=%04x:%02x:%02x.%u\n", selected_pci.domain,
               selected_pci.bus, selected_pci.device, selected_pci.function);
        printf("device_pci_match=%s\n",
               !expected.supplied ||
                       (selected_pci.domain == expected.domain &&
                        selected_pci.bus == expected.bus &&
                        selected_pci.device == expected.device &&
                        selected_pci.function == expected.function)
                   ? "true"
                   : "false");
    } else {
        printf("device_pci_bdf=not-exposed-by-driver\n");
        printf("device_identity=AMD-RX-550-device-name\n");
    }
    printf("device_extension_cl_khr_fp64=%s\n",
           has_extension(selected_extensions, "cl_khr_fp64") ? "true" :
                                                                 "false");
    printf("device_extension_cl_amd_fp64=%s\n",
           has_extension(selected_extensions, "cl_amd_fp64") ? "true" :
                                                                 "false");
    printf("device_double_fp_config=0x%llx\n",
           (unsigned long long)double_fp_config);
    printf("device_int32_host_size=%zu\n", sizeof(cl_int));
    printf("device_global_memory_bytes=%llu\n",
           (unsigned long long)global_memory);
    printf("device_max_allocation_bytes=%llu\n",
           (unsigned long long)max_allocation);
    printf("device_local_memory_bytes=%llu\n",
           (unsigned long long)local_memory);
    printf("device_max_work_group_size=%zu\n", max_work_group_size);
    printf("device_max_work_item_dimensions=%u\n", work_item_dimensions);
    for (cl_uint dimension = 0; dimension < work_item_dimensions; ++dimension) {
        printf("device_max_work_item_size[%u]=%zu\n", dimension,
               work_item_sizes[dimension]);
    }

    if (double_fp_config == 0 ||
        (!has_extension(selected_extensions, "cl_khr_fp64") &&
         !has_extension(selected_extensions, "cl_amd_fp64"))) {
        fprintf(stderr,
                "RESULT=NO_GO\n"
                "Reason: the RX 550 does not report an fp64 extension and "
                "nonzero CL_DEVICE_DOUBLE_FP_CONFIG.\n");
        goto cleanup;
    }
    if (sizeof(cl_int) != 4 || max_allocation < sizeof(double) * SAMPLE_COUNT ||
        global_memory < sizeof(double) * SAMPLE_COUNT ||
        max_work_group_size == 0 || work_item_dimensions == 0 ||
        work_item_sizes[0] == 0) {
        fprintf(stderr,
                "RESULT=NO_GO\n"
                "Reason: one or more required int32, memory or workgroup "
                "limits are insufficient.\n");
        goto cleanup;
    }

    for (i = 0; i < SAMPLE_COUNT; ++i) {
        a[i] = (double)(i + 1) / 10.0;
        b[i] = (double)(i + 2) / 8.0;
        ids[i] = (int32_t)(i * 100003u);
    }

    context = clCreateContext(NULL, 1, &selected_device, NULL, NULL, &status);
    if (status != CL_SUCCESS || context == NULL) {
        print_opencl_error("clCreateContext", status);
        goto cleanup;
    }
    queue = clCreateCommandQueue(context, selected_device, 0, &status);
    if (status != CL_SUCCESS || queue == NULL) {
        print_opencl_error("clCreateCommandQueue", status);
        goto cleanup;
    }

    a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(a), a, &status);
    if (status != CL_SUCCESS || a_buffer == NULL) {
        print_opencl_error("clCreateBuffer(a)", status);
        goto cleanup;
    }
    b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(b), b, &status);
    if (status != CL_SUCCESS || b_buffer == NULL) {
        print_opencl_error("clCreateBuffer(b)", status);
        goto cleanup;
    }
    ids_buffer = clCreateBuffer(context,
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(ids), ids, &status);
    if (status != CL_SUCCESS || ids_buffer == NULL) {
        print_opencl_error("clCreateBuffer(ids)", status);
        goto cleanup;
    }
    sums_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(sums), NULL,
                                 &status);
    if (status != CL_SUCCESS || sums_buffer == NULL) {
        print_opencl_error("clCreateBuffer(sums)", status);
        goto cleanup;
    }
    next_ids_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                     sizeof(next_ids), NULL, &status);
    if (status != CL_SUCCESS || next_ids_buffer == NULL) {
        print_opencl_error("clCreateBuffer(next_ids)", status);
        goto cleanup;
    }

    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL,
                                        &status);
    if (status != CL_SUCCESS || program == NULL) {
        print_opencl_error("clCreateProgramWithSource", status);
        goto cleanup;
    }
    status = clBuildProgram(program, 1, &selected_device, "", NULL, NULL);
    if (status != CL_SUCCESS) {
        size_t log_size = 0;
        char *build_log = NULL;
        clGetProgramBuildInfo(program, selected_device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        if (log_size > 0) {
            build_log = (char *)malloc(log_size + 1);
            if (build_log != NULL &&
                clGetProgramBuildInfo(program, selected_device,
                                      CL_PROGRAM_BUILD_LOG, log_size,
                                      build_log, NULL) == CL_SUCCESS) {
                build_log[log_size] = '\0';
                fprintf(stderr, "OpenCL kernel build log:\n%s\n", build_log);
            }
        }
        free(build_log);
        print_opencl_error("clBuildProgram", status);
        goto cleanup;
    }

    kernel = clCreateKernel(program, "kinera_probe", &status);
    if (status != CL_SUCCESS || kernel == NULL) {
        print_opencl_error("clCreateKernel", status);
        goto cleanup;
    }
    if ((status = clSetKernelArg(kernel, 0, sizeof(a_buffer), &a_buffer)) !=
            CL_SUCCESS ||
        (status = clSetKernelArg(kernel, 1, sizeof(b_buffer), &b_buffer)) !=
            CL_SUCCESS ||
        (status = clSetKernelArg(kernel, 2, sizeof(ids_buffer), &ids_buffer)) !=
            CL_SUCCESS ||
        (status = clSetKernelArg(kernel, 3, sizeof(sums_buffer), &sums_buffer)) !=
            CL_SUCCESS ||
        (status = clSetKernelArg(kernel, 4, sizeof(next_ids_buffer),
                                 &next_ids_buffer)) != CL_SUCCESS) {
        print_opencl_error("clSetKernelArg", status);
        goto cleanup;
    }

    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL,
                                    0, NULL, NULL);
    if (status != CL_SUCCESS) {
        print_opencl_error("clEnqueueNDRangeKernel", status);
        goto cleanup;
    }
    status = clFinish(queue);
    if (status != CL_SUCCESS) {
        print_opencl_error("clFinish", status);
        goto cleanup;
    }
    status = clEnqueueReadBuffer(queue, sums_buffer, CL_TRUE, 0, sizeof(sums),
                                 sums, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        print_opencl_error("clEnqueueReadBuffer(float64)", status);
        goto cleanup;
    }
    status = clEnqueueReadBuffer(queue, next_ids_buffer, CL_TRUE, 0,
                                 sizeof(next_ids), next_ids, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        print_opencl_error("clEnqueueReadBuffer(int32)", status);
        goto cleanup;
    }

    {
        double max_absolute_error = 0.0;
        for (i = 0; i < SAMPLE_COUNT; ++i) {
            const double reference = a[i] + b[i];
            const double absolute_error = fabs(sums[i] - reference);
            const double allowed_error = 1e-12 + 1e-12 * fabs(reference);
            if (absolute_error > max_absolute_error) {
                max_absolute_error = absolute_error;
            }
            if (absolute_error > allowed_error || next_ids[i] != ids[i] + 1) {
                fprintf(stderr,
                        "RESULT=NO_GO\n"
                        "Mismatch at index %zu: float64 gpu=%.17g cpu=%.17g "
                        "absolute_error=%.17g allowed_error=%.17g; "
                        "int32 gpu=%d cpu=%d\n",
                        i, sums[i], reference, absolute_error, allowed_error,
                        next_ids[i], ids[i] + 1);
                goto cleanup;
            }
        }
        printf("sample_count=%d\n", SAMPLE_COUNT);
        printf("float64_cpu_equivalence=PASS\n");
        printf("float64_max_absolute_error=%.17g\n", max_absolute_error);
        printf("float64_atol=1e-12\n");
        printf("float64_rtol=1e-12\n");
        printf("int32_cpu_equivalence=PASS\n");
    }

    printf("h2d_buffers=3\n");
    printf("d2h_buffers=2\n");
    printf("opencl_kernel_build=PASS\n");
    printf("gpu_kernel_execution=PASS\n");
    printf("RESULT=GO_P0_SMOKE_PASS\n");
    exit_code = 0;

cleanup:
    if (next_ids_buffer != NULL) clReleaseMemObject(next_ids_buffer);
    if (sums_buffer != NULL) clReleaseMemObject(sums_buffer);
    if (ids_buffer != NULL) clReleaseMemObject(ids_buffer);
    if (b_buffer != NULL) clReleaseMemObject(b_buffer);
    if (a_buffer != NULL) clReleaseMemObject(a_buffer);
    if (kernel != NULL) clReleaseKernel(kernel);
    if (program != NULL) clReleaseProgram(program);
    if (queue != NULL) clReleaseCommandQueue(queue);
    if (context != NULL) clReleaseContext(context);
    return exit_code;
}
