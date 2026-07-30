// Minimal launcher for AOT-compiled Triton kernels (arms A5a / A5b).
//
// Triton kernels are Python, but Triton emits a cubin, so Python is a
// BUILD-time dependency only. Launching from the same process as every other
// arm keeps an interpreter round-trip out of anything measured, which is the
// confound that makes most published Triton-vs-CUDA numbers hard to trust.
//
// The launch contract is undocumented and was read out of Triton 3.6.0's
// backends/nvidia/driver.py, then validated end to end against
// Python-launched reference values. See docs/TRITON-ABI.md.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OsnTritonKernel OsnTritonKernel;

/// Load `cubin_path` and resolve `kernel_name`. NULL on failure.
OsnTritonKernel* osn_triton_load(const char* cubin_path, const char* kernel_name);
void osn_triton_unload(OsnTritonKernel* k);

/// Launch with `args` as the kernel's own parameters, in declaration order.
///
/// The two trailing scratch pointers Triton's launcher appends
/// (`global_scratch`, `profile_scratch`) are added HERE, so callers pass only
/// the kernel's arguments and cannot forget them. Omitting them does not fail
/// the launch; it reads whatever is in the parameter space.
///
/// `block_dim_x` must be 32 * num_warps from the manifest, NOT the BLOCK
/// constexpr. Returns 0 on success.
int32_t osn_triton_launch(OsnTritonKernel* k, uint32_t grid_x, uint32_t block_dim_x,
                          uint32_t shared_bytes, void** args, int32_t n_args);

#ifdef __cplusplus
}
#endif
