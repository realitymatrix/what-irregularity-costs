# How much of Triton's gap is the tile model?

Asked by a reader of the LinkedIn post: is this mostly a tile versus SIMT
thing, and would a different tile language such as cuTile land closer to
CUDA C++?

The second half needs cuTile. The first half does not, because the tile model's
defining constraint is stateable as arithmetic on probe depths and can be
measured on a run that has already happened.

## The three costs

    per-thread   a lane that resolves retires, so a launch costs the SUM of the
                 lanes' probe depths. This is what CUDA C++ and cuda-oxide pay.
    tile floor   no lane retires until its program is done, so a launch costs
                 the program's DEEPEST lane charged to every lane in it.
    Triton       the trip count is a compile-time constant, so a launch costs
                 that bound charged to every lane, whatever the depths were.

`build/probe_tile_floor` instruments the first two from one run.

## Result

| scene | per-thread | tile floor | tile penalty |
|---|---|---|---|
| sphere, 320k points | 3,826,374 | 6,602,496 | 1.73x |
| sphere, 320k points, r = 2.0 m | 3,793,385 | 7,723,520 | 2.04x |
| RetroOffice P000 frame 8 | 3,644,626 | 6,020,096 | 1.65x |
| AmericanDiner P000 frame 8 | 3,434,197 | 5,126,656 | 1.49x |

**A tile model costs 1.5 to 2x on this workload. Triton costs 18x.**

So the paradigm accounts for under a tenth of the measured gap. The rest is
implementation: a bound that is a compile-time constant rather than a dynamic
block-uniform one, and a compare-exchange that takes no mask, which turns every
wasted iteration into an atomic instead of a no-op.

## What this revises

Our first answer to the question guessed that a different tile language would
beat Triton and still land well short of cuda-oxide. The first half looks
right. The second is probably wrong. A tile language with a dynamic
block-uniform trip count and a maskable compare-exchange has a floor near
1.7x, which is the neighbourhood cuda-oxide occupies at 1.2 to 1.4x, not the
neighbourhood Triton occupies at 18x.

The honest statement is therefore narrower than the one we published. Triton's
cost on irregular work is not evidence that tile models cannot express
irregular work. It is evidence that Triton's particular API cannot, and the
two are easy to conflate when only one tile language has been measured.

## What this is not

It prices work, not time. It counts probe steps and assumes time follows them,
which ignores memory behaviour, occupancy and code generation. It also assumes
perfect aggregation, a dynamic bound and a maskable exchange, none of which any
shipping tile language is confirmed to have here.

It is deliberately generous to the tile penalty in one respect. The instrument
takes the maximum over each thread's *total* steps, so it models a program that
runs one uniform loop nest to its slowest lane. A tile implementation that
aggregates per call site would pay the sum of per-call maxima, which is smaller.
The floor reported here is therefore an over-estimate of the tile penalty, and
the real one is lower still.

So it is a floor, not a prediction. An arm lands above its own floor. If a real
cuTile arm lands below these numbers the model is wrong, which is worth knowing
either way.

## Testing it properly

A cuTile arm is possible. cuTile shipped in CUDA 13.1 with the Python DSL in
13.2, and this machine is on 13.0, so it needs a toolkit upgrade first. That
upgrade is not free: the published measurements state CUDA 13.0, so every arm
would have to be re-run on 13.2 to stay comparable. The arms sit behind a C ABI,
so the arm itself is the small part.

## Reproducing

```bash
build/probe_tile_floor data/tartanair/retro_p000_f008.bin
```
