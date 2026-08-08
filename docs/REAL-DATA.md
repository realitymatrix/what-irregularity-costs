# Regenerating the real-data cells

`data/` is not in git: the source frames are gigabytes and belong to TartanAir,
not to this repo. What is in git is everything needed to rebuild the clouds
byte for byte, which is the part that matters for checking a result.

## What the cells are

Eleven cells on the `real` axis, from TartanAir V2, `Data_easy`, `lcam_front`,
ground-truth depth unprojected at full resolution.

| Cells | Source | What they vary |
|---|---|---|
| `tartan-cold`, `tartan-warm` | RetroOffice P000 frame 8 | cold vs warm volume, same frame |
| `tartan-p001` … `p005` | RetroOffice P001–P005 frame 8 | camera path, same room |
| `tartan-dense` | RetroOffice P000 frame 8 | pool sized to the block count, not the scene |
| `diner-p000`, `p002`, `p003` | AmericanDiner P000/P002/P003 frame 8 | a different interior |

One frame of one trajectory is a sample of size one, and the real-data number
is the one a practitioner will actually act on, so it is the one that least
deserves to rest on a single scene. The trajectories vary the camera path while
holding the room fixed; AmericanDiner varies the room.

## Fetching

RetroOffice was fetched with the `tartanair` pip package. AmericanDiner was
pulled straight from the Hugging Face mirror, which is lighter — one modality
of one environment rather than the package's defaults:

```python
from huggingface_hub import hf_hub_download
hf_hub_download("theairlabcmu/tartanair2",
                "AmericanDiner/Data_easy/depth_lcam_front.zip",
                repo_type="dataset", local_dir=".")
```

The zip contains `pose_lcam_front.txt` per trajectory, so depth alone is
enough; `pose` is not a valid download modality and asking for it fails.

Sizes vary by two orders of magnitude across environments, so check before
fetching: AmericanDiner is 366 MB, ConstructionSite 2.7 GB, GreatMarsh 21 GB.

## Converting

`tools/tartanair_points.py` needs `cv2` — not an incidental dependency. The
depth is float32 packed into a BGRA PNG, and PIL reorders the channels, which
silently produces plausible-looking wrong depth.

```bash
python tools/tartanair_points.py <env>/Data_easy/P000 --verify   # check the pose convention
python tools/tartanair_points.py <env>/Data_easy/P000 \
    --frame 8 --out data/tartanair/<tag>_p000_f008.bin \
    --warm-out data/tartanair/<tag>_p000_f008_warm.bin --warm-frames 8
```

Run `--verify` on any new environment before trusting a cloud from it. The
pose convention is NED and getting it wrong does not raise: it produces a
cloud that looks fine frame by frame and disagrees between frames. The check
fuses consecutive frames and watches how the occupied-block count grows, which
separates the right permutation from the wrong ones by an order of magnitude
without needing a tuned threshold. It passed on all trajectories used here.

## Two harness traps found while adding these cells

**A cell that sets `pool_factor` and a warm cloud used to size its pool from
the frame alone.** The warm pre-fill then overflowed the pool before the timed
window opened, so the cell measured pool exhaustion rather than table
contention, and an unbounded probe against a full table scans every slot for
every point — a cell that should take seconds took over half an hour. The
probe now integrates the warm cloud too.

**A sweep with a missing arm used to run to completion and report "0
invalid".** `OSN_TRITON_DIR` is resolved relative to the working directory, and
`sweep.sh` runs from the build directory, so an override that looks right from
the repo root silently drops the Triton arm. The harness now refuses to run
unless `OSN_ALLOW_MISSING_ARMS=1` says the omission was intended. This is the
same failure as the fake device axis in the paper: the run completes, the
numbers look ordinary, and nothing downstream can tell that a third of the
comparison is absent.
