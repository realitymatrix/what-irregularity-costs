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
