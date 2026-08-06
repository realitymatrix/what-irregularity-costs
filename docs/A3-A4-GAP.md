# Attributing the A3 / A4 performance gap

Batched timing (see bench/bench_arms.cu) puts the CUDA C++ arm ahead of the
Rust arm on both integrate stages, p50 in ms:

| stage | A3 cuda | A4 rust | ratio |
|---|---|---|---|
| allocate | 0.028 | 0.046 | 1.64x |
| update | 0.249 | 0.303 | 1.22x |

Both arms implement the same algorithm over the same memory and emit the same
PTX instruction family for the CAS and the atomic adds, so the gap is a
code-generation result and worth attributing rather than reporting.

## Refuted: register pressure

The obvious first hypothesis, and wrong. A4 uses **fewer** registers:

| kernel | A3 | A4 |
|---|---|---|
| allocate | 34 | 28 |
| update | 40 | 34 |

Neither arm spills. Lower register use means A4 has at least as much occupancy
headroom, so occupancy cannot explain it being slower.

Reading A4's register counts is awkward: `cuda-oxide` embeds the device image in
the host object rather than emitting a cubin, so `cuobjdump` finds nothing in
the archive. Extract with `ar x`, locate the embedded ELF inside
`*.embed.o` (it starts at a non-zero offset), and dump that.

## Established: A4's update executes 62% more instructions

SASS instruction counts for the update kernel:

    A3 (CUDA C++)   312 instructions
    A4 (Rust)       504 instructions   1.62x

The largest opcode deltas, A4 minus A3:

    LOP3   +27      logic ops
    FFMA   +22
    FSETP  +22      float predicate sets
    BRA    +21      branches
    ISETP   +7
    IMAD   -10

A 1.62x instruction count against a 1.22x runtime is consistent with a kernel
that is partly bound by its atomics rather than by issue rate: extra
instructions cost, but less than proportionally.

## Refuted: `read_volatile` on the hash key

A4 read the hash key with `read_volatile()` where A3 uses a plain load,
reserving `volatile` for the `block_idx` spin. That looked decisive: the probe
loop is the hottest path in the allocate kernel, and an uncached load there
would cost exactly the kind of time the gap represents.

Changing it to a plain `read()` produced **byte-identical SASS**: 528
instructions before and after, with no opcode counts changed at all. LLVM and
NVVM generate the same code either way here, so the marking was never the
cause. The change is kept because a plain read is the correct expression of the
intent, but it buys nothing.

## Refuted: `f32::clamp` NaN semantics

Rust's `clamp` is NaN-propagating and asserts its bounds, where CUDA's
`fminf(fmaxf(...))` is a bare pair of selects, so it looked like a plausible
source of the FSETP delta. Replacing it with a hand-rolled branchless select
changed the instruction count by **zero** (504 to 504); one `FMNMX` became one
`FSEL`. The hypothesis is dead.

## Open: allocate is 1.64x slower on nearly identical instruction counts

    A3 allocate   520 instructions
    A4 allocate   528 instructions   1.02x

Instruction count explains the update gap but explicitly does NOT explain the
allocate gap. Something else dominates there. Candidates not yet tested:

* **The publication spin.** Both arms spin on `block_idx` while a peer
  publishes it, but the loop bodies differ and divergent waiting is sensitive
  to how the compiler schedules the reload.
* **Memory access patterns.** Same addresses, but different coalescing or
  different scheduling of the probe loads.
* **Warp divergence.** A4 has +8 branches in allocate; if they sit inside the
  probe loop the cost is multiplied by probe depth.

### ncu is blocked on this machine

`ncu` is installed but refuses with `ERR_NVGPUCTRPERM`: access to GPU
performance counters is restricted to administrators. Lifting it needs root
either way:

    # /etc/modprobe.d/nvidia-profiling.conf
    options nvidia NVreg_RestrictProfilingToAdminUsers=0
    # then reboot

Until then, warp stall reasons, memory throughput and achieved occupancy are
all unavailable, and the attribution is limited to static analysis of the
generated code.

### Suggestive, not conclusive: memory-ordering scope

The SASS memory instructions differ in scope, and the scopes are not cosmetic:
`STRONG.SYS` is a system-scope strongly-ordered access, which is materially
more expensive than a plain or GPU-scope one.

| kernel | plain LDG.E | LDG.E.SYS | LDG.E(.64).STRONG.SYS | atomics |
|---|---|---|---|---|
| A3 allocate | 0 | 6 | 2 | 10 |
| A4 allocate | 3 | 0 | 6 | 7 |

A4 issues **six** system-scope strongly-ordered loads against A3's **two**,
while issuing *fewer* atomics. That is the clearest asymmetry found so far and
it points at how `cuda-oxide` lowers Rust's volatile and atomic accesses versus
how nvcc lowers CUDA's `volatile` qualifier.

It is not proof. Counting instructions says nothing about how often each is
executed or how long each stalls, and the allocate kernel's cost is dominated
by contention that a static count cannot see. Confirming it needs the counters
above, or a variant experiment that removes the remaining `read_volatile` calls
entirely and accepts the correctness loss to isolate their cost.

## What this does and does not license saying

Established: A4 is slower on both stages under batched timing; it is not a
register-pressure or occupancy effect; the update gap tracks a 1.62x
instruction-count difference; `clamp` is not the cause.

Not established: why Rust-to-PTX emits 62% more instructions for the same
source algorithm, and why allocate is slower at parity of instruction count.
Neither should be claimed until measured.
