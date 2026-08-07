# `DeviceAtomic*::load` and `::store` cannot be called

Found 2026-08-07 while trying to close arm A4's unattributed allocate gap.
Upstream is `NVlabs/cuda-oxide` at `20a5616`; the patch is on a local branch
`scoped-atomics`.

## What this corrects

This project previously recorded two claims about `cuda-oxide` that were
wrong, and one that was right for the wrong reason.

**Wrong: "cuda-oxide gives no control over memory scope."** It does.
`cuda-device` ships `DeviceAtomic*`, `BlockAtomic*` and `SystemAtomic*` types
with a documented scope table mapping to PTX `.gpu`, `.cta` and `.sys`. The
`core::sync::atomic` types are separate and are hardcoded to system scope at
`crates/mir-importer/src/translator/terminator/intrinsics/atomic.rs:813`
(`// core atomics always use system scope`), which is a defensible
conservative default, not a missing feature.

**Wrong: "libNVVM rejects atomic loads and stores, so the language forces
`read_volatile`."** libNVVM does reject them, but the conclusion was
backwards: `cuda-oxide` exposes scoped atomic `load` and `store` precisely so
that callers do not need volatile. The problem is that those methods do not
work.

**Right, but not for the stated reason: A4 ends up issuing system-scope
strongly-ordered loads.** It does, because `read_volatile` is what remains
after `load` fails, and Rust's volatile semantics lower to a system-scope
access. The cause is the broken API, not an absent one.

## The defect

`DeviceAtomicI32::load(AtomicOrdering::Relaxed)` compiles under the default
LLVM path and fails under `--materialize-cubin`:

    error: [rustc_codegen_cuda] Failed to embed device artifact:
           libnvvm error in nvvmVerifyProgram: 6
      --- libNVVM log ---
      error: Function `..._find_block' Basic Block `bb5':
        context:   %v48 = load atomic i32, ptr %v47 syncscope("device") monotonic, align 4
        Atomic loads/stores are not supported

Every call fails. The methods are documented public API with specified
orderings (`load` accepts Relaxed/Acquire/SeqCst, `store` accepts
Relaxed/Release/SeqCst), the NVVM dialect models them as `NvvmAtomicLoadOp`
and `NvvmAtomicStoreOp`, and `mir-lower` lowers both to `llvm::AtomicLoadOp` /
`llvm::AtomicStoreOp` — the one IR form libNVVM refuses.

Three things make this worse than a missing feature:

1. **It is unusable in every configuration that matters.** Anything built with
   `--materialize-cubin`, which is any real kernel, cannot call these methods.
2. **The failure is a backend verifier message, not a call-site error.** It
   names an LLVM basic block, not the line of Rust that caused it.
3. **The fallback is silently more expensive.** With `load` unavailable the
   only remaining way to re-read a location another thread is publishing to is
   `read_volatile`, and that lowers to a system-scope strongly-ordered access
   — ordering against the host and peer devices for data that never leaves one
   device's global memory.

## The fix

The instructions exist; only the IR-level form is unsupported. PTX has had
`ld.relaxed.gpu`, `ld.acquire.gpu`, `st.relaxed.gpu` and `st.release.gpu`
since sm_70. So `convert_atomic_load` and `convert_atomic_store` in
`crates/mir-lower/src/convert/intrinsics/atomic.rs` now emit inline PTX
instead, following the pattern `convert_packed_atom_add` in the same file
already uses for packed atomic add:

    ld.{sem}.{scope}.{ty} $0, [$1];     constraints  "={reg},l,~{memory}"
    st.{sem}.{scope}.{ty} [$0], $1;     constraints  "l,{reg},~{memory}"

with `{scope}` from the op's existing `AtomicScope` attribute (`gpu` / `cta` /
`sys`), `{ty}` and `{reg}` from operand width (`b16`/`h`, `b32`/`r`,
`b64`/`l`), and `{sem}` from the ordering.

Two deliberate choices:

* **`AsmKind::SideEffect` with a `~{memory}` clobber.** A scoped atomic load is
  usually a spin on another thread's publication, so it must be re-issued each
  iteration rather than hoisted out of the loop. This is the property
  `read_volatile` was being used for.
* **`SeqCst` is rejected with an explanatory error rather than approximated.**
  A plain PTX load or store cannot express sequential consistency; it needs a
  `fence.sc`. Emitting `relaxed` where `seq_cst` was requested would be a
  correctness bug that no test would catch. Acquire-on-store and
  release-on-load are rejected the same way.

## Status: the patch works, and the hypothesis it was built to test is dead

* `mir-lower` compiles, the emitted PTX assembles, and `cargo oxide build
  --materialize-cubin` succeeds on a kernel that previously failed the libNVVM
  verifier. Point the build at it with
  `CUDA_OXIDE_BACKEND=<worktree>/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so`.
* **The SASS carries GPU scope.** Arm A4's allocate kernel went from six
  `LDG.E(.64).STRONG.SYS` to zero, with all seven atomics already
  `STRONG.GPU`. Correctness is unchanged: 1160 blocks, mean surface distance to
  the CUDA C++ arm still 0.000000000 m.
* **A4's numbers did not move.** 0.0460 to 0.045 ms at the baseline cell and
  0.0783 to 0.078 on the narrow card, which is noise. Memory-ordering scope is
  therefore refuted as the cause of the allocate gap (docs/A3-A4-GAP.md).

That is a negative result for the performance hypothesis and a positive one for
the patch: the experiment was impossible to run before, and it is now cheap to
run. The defect is worth reporting upstream on its own merits.
* **Submitted upstream as NVlabs/cuda-oxide#695** (draft), 2026-08-07:
  https://github.com/NVlabs/cuda-oxide/pull/695
* Rebased onto upstream `main` after it moved 65 commits ahead. The conflict
  was two sets of tests appended to the same end-of-file; resolving it by
  concatenating both sides of each hunk produced a file that did not compile,
  because one hunk had split an upstream test in half. Taking upstream's file
  wholesale and re-appending the two new tests is the correct resolution and
  the one to reach for whenever a conflict is append-versus-append.
* Reviewing the change against CONTRIBUTING.md found three things worth
  recording, because none of them would have failed locally:
  - The new example carried an **NVIDIA copyright header** copied from
    `vecadd`. CONTRIBUTING says to name a copyright holder only when you are
    one or are authorised to; the file now carries the SPDX licence identifier
    alone.
  - The example needed registering in `NVVM_VERIFY_EXAMPLES` in
    `scripts/smoketest.sh`. That array is what runs an example through the real
    libNVVM verifier in compile-only mode, which is the only CI lane that has
    no GPU. Without the entry the example still runs, but the regression this
    whole change exists to prevent would be invisible on exactly the runners
    most likely to catch it.
  - Their process asks for an **issue before the PR**. Not done; the PR is
    otherwise complete.
* Scope grew by one during the work. The shipped `atomics` example turned out
  not to build under `--materialize-cubin` either, failing first on
  `fence syncscope("block") release` -> "Illegal instruction: fence" in
  `atomic_block_scope_acqrel_test`. That is the same defect class (an IR
  construct libNVVM refuses where the PTX instruction exists), so `emit_fence`
  is fixed the same way. Without it the example could not be made to build and
  the fix could not be demonstrated end to end.
* Superseded: upstream tests. `crates/dialect-nvvm/tests/ops_test.rs`
  already exercises `NvvmAtomicLoadOp`/`NvvmAtomicStoreOp` construction, but
  the lowering now also has two tests asserting the emitted templates,
  constraints and asm kind, and that inexpressible orderings are rejected. A
  new `scoped_atomic_load_store` example covers the path end to end: it fails
  to compile on upstream `main` and passes with the change, which is what makes
  it a regression test rather than a demo.

## Why this is worth reporting upstream regardless of the measurement

Whether or not it closes A4's gap, an API that cannot be called is a defect,
and this one is invisible until a user reaches `--materialize-cubin`. It also
explains why a Rust GPU kernel would look worse than its CUDA C++ counterpart
on memory-ordering-sensitive code, which is exactly the comparison NVIDIA's
own documentation says it wants to make and whose appendix is still an empty
placeholder.
