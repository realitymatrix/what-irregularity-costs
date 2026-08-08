# What Irregularity Costs

CUDA C++, Rust, and Triton on a hash-blocked GPU workload.

**[Paper (PDF)](paper/main.pdf)** &middot; [Project page](docs/index.html) &middot;
[Raw data](results/)

Submitted to arXiv, primary cs.CV, cross-listed cs.PL, cs.DC and cs.PF. Under
moderation; this line gets the identifier once it is announced. The submission
package and the script that builds and verifies it are in [arxiv/](arxiv/).

GPU language comparisons are almost always run on tiled dense linear algebra,
where every toolchain is good and the differences are small. This repository
implements the same hash-blocked TSDF fusion kernel five times and measures it on
a workload with the opposite character: an open-addressed hash table with
compare-exchange insertion, data-dependent per-lane probe depth, and contended
scatter.

The result is a split. On the regular stage all three languages land within a
small factor of each other. On the irregular stage Rust stays close to
hand-written CUDA C++ and Triton is more than an order of magnitude behind.

| stage | Rust / CUDA C++ | Triton / CUDA C++ |
|---|---|---|
| allocate (irregular) | 1.02–3.34x | 11.2–31.6x |
| update (regular) | 0.96–1.19x | 1.1–2.6x |

Full integrate path on real depth data: Rust 1.01–1.03x, Triton 2.57–2.92x.

## What is here

| path | |
|---|---|
| `cpp/` | CUDA C++ and CPU arms, the measurement harness, the correctness gates |
| `crates/tsdf-rust-cuda/` | Rust arm, compiled to PTX through `cuda-oxide` |
| `triton/`, `tools/` | Triton arm and its ahead-of-time compilation |
| `paper/` | LaTeX source; every table is generated from `results/` |
| `results/` | Raw measurement CSVs behind the paper's tables |
| `docs/` | Design records, including how each finding was reached and corrected |

## Reproducing

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
build/test_cuda_analytic          # correctness gate: all arms vs analytic geometry
tools/sweep.sh build              # the full matrix, 3 passes per device
paper/gen_tables.py  results/sweep_*/sweep.csv  # paper tables
tools/gen_figures.py results/sweep_*/sweep.csv  # paper figures + page charts
tools/gen_page.py    results/sweep_*/sweep.csv  # docs/index.html
```

Nothing in the paper or on the project page is typed in by hand: the tables, the
figures and the page all come off those three generators reading the same CSVs,
so a number cannot drift in one place without moving in the others.

Requires an NVIDIA GPU. Measurements in the paper are sm_120 (Blackwell), CUDA
13.0, driver 580.

**The Rust arm needs `cuda-oxide` at or after the atomics fix.** Its scoped
atomic load and store could not be compiled at all in the version used for these
measurements. The fix,
[NVlabs/cuda-oxide#695](https://github.com/NVlabs/cuda-oxide/pull/695), is
merged upstream, so a current checkout works without patching. On an older one,
build the backend from that commit and point `CUDA_OXIDE_BACKEND` at the
resulting `librustc_codegen_cuda.so`. See
[docs/CUDA-OXIDE-ATOMIC-LOAD.md](docs/CUDA-OXIDE-ATOMIC-LOAD.md).

## A note on the docs

`docs/` records how the work actually went, including figures that were published
and later corrected: Triton's gap reported as 73x and corrected to 17.9x, a
device axis that turned out to be measuring one GPU twice, and a residual
attributed to the wrong cause. They are kept because a comparison of this kind is
only as good as its account of what it got wrong.

## Licence

MIT. See [LICENSE](LICENSE).
