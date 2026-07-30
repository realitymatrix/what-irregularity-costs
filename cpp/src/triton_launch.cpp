#include "osn_tsdf/triton_launch.h"

#include <cuda.h>

#include <cstdio>
#include <fstream>
#include <vector>

struct OsnTritonKernel {
    CUmodule module = nullptr;
    CUfunction func = nullptr;
    // Scratch pointers Triton's launcher passes after the kernel's own args.
    // Null when the kernel declares no scratch, which is what the Python
    // launcher passes in that case.
    CUdeviceptr global_scratch = 0;
    CUdeviceptr profile_scratch = 0;
};

namespace {
bool check(const char* op, CUresult r) {
    if (r == CUDA_SUCCESS) return true;
    const char* name = nullptr;
    cuGetErrorName(r, &name);
    std::fprintf(stderr, "[triton_launch] %s failed: %s\n", op, name ? name : "?");
    return false;
}
}  // namespace

extern "C" {

OsnTritonKernel* osn_triton_load(const char* cubin_path, const char* kernel_name) {
    std::ifstream f(cubin_path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[triton_launch] cannot open %s\n", cubin_path);
        return nullptr;
    }
    std::vector<char> image((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
    if (image.empty()) return nullptr;

    // The context is created by whoever runs first; the CUDA runtime's primary
    // context is already current by the time the arms run, so no context is
    // created here. Creating one would put driver allocations in a different
    // context from the runtime allocations the C++ arms make.
    auto* k = new OsnTritonKernel();
    if (!check("cuModuleLoadData", cuModuleLoadData(&k->module, image.data())) ||
        !check("cuModuleGetFunction", cuModuleGetFunction(&k->func, k->module, kernel_name))) {
        delete k;
        return nullptr;
    }
    return k;
}

void osn_triton_unload(OsnTritonKernel* k) {
    if (!k) return;
    if (k->module) cuModuleUnload(k->module);
    delete k;
}

int32_t osn_triton_launch(OsnTritonKernel* k, uint32_t grid_x, uint32_t block_dim_x,
                          uint32_t shared_bytes, void** args, int32_t n_args) {
    if (!k || !k->func || n_args < 0) return 1;
    std::vector<void*> params;
    params.reserve(static_cast<std::size_t>(n_args) + 2);
    for (int32_t i = 0; i < n_args; ++i) params.push_back(args[i]);
    // driver.py:261-262: global_scratch then profile_scratch, always present.
    params.push_back(&k->global_scratch);
    params.push_back(&k->profile_scratch);

    const CUresult r = cuLaunchKernel(k->func, grid_x, 1, 1, block_dim_x, 1, 1, shared_bytes,
                                      nullptr, params.data(), nullptr);
    return check("cuLaunchKernel", r) ? 0 : 2;
}

}  // extern "C"
