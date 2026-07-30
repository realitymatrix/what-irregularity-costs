# Triton launch ABI, and why the Rust driver launches Triton kernels directly

## Why not embedded Python

The Triton fusion arm has to run under the same Rust driver as every other arm.
Launching `@triton.jit` kernels from embedded CPython would put an interpreter
round-trip inside the measured path, so the benchmark would partly be measuring
language runtime rather than kernel time. That is the confound which makes most
published "Triton vs CUDA" comparisons hard to trust, and it is avoidable:
Triton exposes the compiled cubin.

So `tools/triton_aot.py` compiles the kernels and emits `<name>.cubin` plus a
`<name>.json` manifest at **build time**, and `crates/triton-aot` loads and
launches them through the CUDA driver API. Python never appears at runtime.

State this explicitly in the paper. It is a methodological strength, not an
implementation detail.

## The ABI

Undocumented. Read out of `triton/backends/nvidia/driver.py`, Triton 3.6.0:

| Aspect | Rule | Source |
|---|---|---|
| Parameters | non-constexpr args in declaration order, then `global_scratch`, then `profile_scratch` | `driver.py:261-262` |
| Block shape | `blockDimX = 32 * num_warps`, Y = Z = 1 | `driver.py:330-332` |
| Shared memory | `metadata.shared`, passed as dynamic shared | `driver.py` launch config |
| Constexprs | compiled into the cubin, never passed | `src.signature` |

Two traps worth naming:

1. **`BLOCK` is not the CUDA block size.** `BLOCK` is a Triton tile size. The
   CUDA block comes from `num_warps`. In these kernels `BLOCK = 256` but
   `blockDimX = 128`.
2. **The two trailing scratch pointers are mandatory even when their sizes are
   zero.** The kernel still declares the parameters, so omitting them leaves
   the launcher reading whatever is in the param space.

Observed on the S2 hash-insert kernel: signature declares 4 runtime args, the
PTX `.visible .entry` declares 6 params. The difference is those two.

## Validation, and validation of the validation

Every one of these mistakes fails **silently**: the launch returns success and
the kernel computes the wrong answer. So `validate_abi` checks numerical output
against values produced by the Python-launched kernels
(`artifacts/triton/expected.json`), rather than checking that the launch
succeeded.

```
=== ABI VALIDATION: PASS ===
  winners 64 / unresolved 0 / distinct slots 64 / scratch intact
  total count 512 / weight 256 / sum_x 512 / sum_z 1536
```

A test that cannot fail is worthless, so both failure modes were deliberately
injected and confirmed detected:

| Injected fault | Result |
|---|---|
| Trailing scratch params omitted | **SIGSEGV**, exit 139 |
| `BLOCK` (256) used as block size instead of `32*num_warps` (128) | **FAIL**, exit 1 |

The second is the dangerous one: it runs to completion and produces wrong
numbers. It is caught only because the checks are numerical.

## On upgrading Triton

This is not a stable public interface. After any Triton bump:

1. Re-read `driver.py` for the param order and block-shape rules.
2. Re-run `tools/triton_aot.py`.
3. Re-run `validate_abi`. It must PASS before any timing is trusted.
