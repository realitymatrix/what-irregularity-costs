# Phase 5: the depth stage

Architecture decided 2026-07-30: **Rust drives pre-built TensorRT engines, with
device-resident I/O.** Depth never round-trips to host; it feeds the TSDF
integrate path directly, which is what makes the host-transfer ablation a real
measurement rather than an afterthought.

## Constraint: the runner is written here, not reused

OSN's `libinfer` is the reference for the *architecture* (load a pre-built
engine, expose device-pointer inference) but not a source of code. Under the
no-reuse rule this project writes its own TensorRT runner. Two things carry
over as design decisions rather than as text:

* **The runner does not build engines.** Engine construction is a separate,
  slow, machine-specific step done with `trtexec`; the runtime only loads.
  Mixing the two puts minutes of autotuning inside a process that should start
  in milliseconds.
* **Device-pointer inference is the primary entry point**, not an optimisation.
  A host round-trip between depth and fusion would be an artefact of the
  harness rather than of the pipeline.

## Models

Both arms of the crossover claim are present locally.

| | FoundationStereo | Fast-FoundationStereo |
|---|---|---|
| checkpoint | `23-51-11` (vit-large) | `23_36_37`, also `20_26_39`, `20_30_48` |
| size | 718 MB | 67 MB (iters 4) / 77 MB (iters 8) |
| **ONNX nodes** | **55,047** | **4,872** |
| opset | 16 | 17 |
| input | `left`, `right` `[batch,3,480,640]` | `left_image`, `right_image` `[1,3,320,736]` |
| output | `disp` `[batch,1,H,W]` | `disparity` `[1,1,320,736]` |
| batch | dynamic | static |

An 11x difference in node count between the accuracy arm and the speed arm is
the whole basis of the crossover: with the heavyweight model in front, fusion
cost is irrelevant; with the real-time one, fusion decides frame rate.

Note the resolutions differ from the roadmap's assumption. 576x960 and 320x736
are **Fast-FoundationStereo's** deployable shapes; the FoundationStereo export
here is 480x640 with a dynamic batch dimension.

Checkpoint choice matters: `23-51-11` is the vit-large model. An earlier
benchmark in the OSN work was accidentally run against `11-33-40`, a vit-small
checkpoint, which is not comparable.

## Open design decision: resolution mismatch

TartanAir V2 frames are **640x640**. Neither model accepts that shape:

    FoundationStereo        480x640   height must shrink by 160
    Fast-FoundationStereo   320x736   height must shrink by 320, width GROW by 96

Resizing across aspect ratio is not an option. Stretching a stereo pair changes
the horizontal scale, and disparity is a horizontal measurement: `Z = fx*B/d`
stops holding the moment `fx` and the image width disagree. It also degrades
the depth itself, which is a documented failure in the prior work.

The candidates, none free:

1. **Centre-crop vertically.** 640x640 to 480x640 keeps `fx`, `fy` and `cx`
   exactly; only `cy` shifts by 80 px. Clean for FoundationStereo, and the
   arithmetic stays valid. Costs 25% of vertical field of view.
2. **Crop then pad** for Fast-FoundationStereo: crop 640 to 320 vertically,
   pad 640 to 736 horizontally with a border. Padding is safe for `fx` but
   introduces a 96 px region with no real content, where disparity is
   meaningless and must be masked before backprojection.
3. **Re-export the ONNX at 640x640.** Removes the problem entirely and keeps
   the intrinsics untouched. Costs an export step per model, and the
   FoundationStereo graph is 55k nodes, so the export is not trivial.

Option 1 for FoundationStereo and option 3 for Fast-FoundationStereo is the
likely answer, but this is a decision to make deliberately rather than by
default, because every option changes what the depth-accuracy numbers mean.

## Engine build

Engines are architecture-specific and are not committed. Build with:

    trtexec --onnx=<model>.onnx --saveEngine=<out>.engine \
            --fp16 --builderOptimizationLevel=3 --memPoolSize=workspace:4096 \
            --minShapes=left_image:1x3x320x736,right_image:1x3x320x736 \
            --optShapes=... --maxShapes=...

FP16 throughout, matching how both models are characterised upstream. The
engine must be built on the same architecture it runs on: a PTX-JIT fallback
would reintroduce exactly the confound the fusion arms were careful to avoid.


## Both depth arms measured

Fast-FoundationStereo, iters=4, 320x736, FP16, sm_120, engine built locally:

    p50 13.358 ms   p95 13.708 ms   p99 13.854 ms   min 13.042 ms   (n=50)

Against the fusion arms (allocate + update, batched, same GPU):

| arm | fusion ms | depth ms | fusion share |
|---|---|---|---|
| A3 cuda | 0.277 | 13.358 | **2.0%** |
| A4 rust | 0.347 | 13.358 | 2.5% |
| A5 triton | 2.392 | 13.358 | 15.2% |

**This is evidence against the project's headline claim.** The crossover was
stated as: with a heavyweight depth model the fusion backend is irrelevant, but
swap in a real-time model and fusion becomes the bottleneck. Here the
*real-time* model already costs 48x the fastest fusion arm. Fusion is 2% of the
pipeline, and the entire spread between the best and worst fusion arm is 15% of
end-to-end.

Caveats, none of which look large enough to rescue the claim as stated:

* iters=4 is the fastest Fast-FoundationStereo variant; iters=8 is slower.
* 320x736 is 235k pixels against TartanAir's 640x640 = 410k, so fusion at
  matched resolution would be somewhat *cheaper*, not dearer.
* The 13.4 ms includes a host synchronise, worth roughly 0.13 ms at worst.
* FoundationStereo, the accuracy arm, has 11x the node count and will be far
  slower still, pushing fusion's share toward noise.

So the honest reading is that a crossover does not occur at this scale, and the
paper should not promise one. Three ways forward, in order of how much they
preserve the original framing:

1. **Find where the crossover actually is, if anywhere.** Fusion cost scales
   with points and scene extent; depth cost is fixed per frame. Multi-frame
   sequences with a growing volume, larger resolutions, and the extraction and
   eviction stages all push in fusion's favour. Extraction alone is already
   1.8 ms, an order of magnitude above integrate.
2. **Reframe as a negative result.** "Fusion backend choice is worth 2% of a
   stereo reconstruction pipeline, and here is the measurement that shows it"
   is a defensible systems finding, and more useful to a practitioner than a
   crossover that only appears under contrived conditions.
3. **Keep the language comparison as the primary contribution.** A3 vs A4 vs A5
   stands on its own: it is the comparison NVIDIA has not published, it is
   backed by correctness triangulation, and it does not depend on the depth
   stage at all.

None of these is chosen yet. The decision needs the FoundationStereo engine and
a multi-frame sequence before it is well-founded, since a single frame of one
model at one resolution is a thin basis for abandoning the framing.


## FoundationStereo, and the crossover verdict

FoundationStereo `23-51-11`, 480x640, FP16, sm_120. The build expanded 55,047
ONNX nodes into **134,078 TensorRT layers** and took 258 s, producing an
818 MB engine.

    p50 157.236 ms   p95 157.576 ms   p99 157.608 ms   min 155.983 ms   (n=30)

### The two arms, against fusion

| depth model | depth | best fusion share | fusion arm spread |
|---|---|---|---|
| Fast-FoundationStereo, iters 4, 320x736 | 13.4 ms | 2.03% | 15.5% |
| FoundationStereo, 23-51-11, 480x640 | 157.2 ms | **0.18%** | 1.3% |

Depth ratio between the arms is **11.8x**, against an ONNX node ratio of
11.3x. Runtime tracks graph size almost exactly, which is a tidy sanity check
on both engines: neither is accidentally falling back to a slow path.

### Verdict: the crossover does not occur

The claim was that swapping a heavyweight depth model for a real-time one makes
fusion the bottleneck. Measured, at both ends of an 11.8x depth range:

* With the heavyweight model, fusion is **0.18%** of the pipeline and the
  entire spread between the best and worst fusion backend is **1.3%**.
* With the real-time model, fusion is **2.0%** and the spread is 15.5%.

Fusion never approaches being the bottleneck. The claim as written is not
supported, and no amount of caveat-hunting rescues it: the fastest available
depth model, at fewer pixels than the dataset's native resolution, still costs
48x the fastest fusion arm.

Two things this does NOT invalidate:

1. **The language comparison.** A3 vs A4 vs A5 is measured, correctness-gated
   and independent of the depth stage. It remains the strongest contribution,
   and it is still the comparison NVIDIA has not published.
2. **The result itself.** "Fusion backend choice is worth between 0.2% and 2%
   of a stereo reconstruction pipeline" is a useful engineering finding,
   arrived at honestly. It tells a practitioner where not to spend effort,
   which is what a latency-budget paper should do.

### What could still move the numbers

These are worth measuring before the framing is finalised, not because they are
likely to rescue the crossover but because they bound it:

* **Extraction, not integrate.** Extraction is 1.825 ms, 6.6x the whole
  integrate path. Over a sequence it runs once rather than per frame, but on a
  budget where fusion totals 2 ms it dominates the fusion side.
* **Scene growth.** Fusion cost scales with points and allocated blocks; depth
  is fixed per frame. A long sequence with a large volume shifts the ratio, and
  a single sphere at 320k points is the small end.
* **A faster depth stage than exists.** The gap is 48x. Closing it needs a
  depth model roughly two orders of magnitude cheaper than FoundationStereo,
  which is a different research problem, not a tuning exercise.
