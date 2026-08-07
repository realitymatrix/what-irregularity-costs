#!/usr/bin/env python3
"""Unproject TartanAir V2 frames into world-space point clouds for the sweep.

Every scene the arms have been measured on is synthetic: a sphere or a plane.
Those have uniform point density, a single convex surface, and a hash occupancy
that a formula predicts. Real depth has none of that, and the parts of the
algorithm under test (probe depth, per-lane divergence, atomic contention) are
exactly the parts that depend on how points cluster.

Output is a flat binary the C++ harness mmaps, rather than anything requiring a
parser on the C++ side:

    magic   char[4]   "OSNP"
    version int32     1
    n       int32     point count
    cam     float32[3] camera centre in world coordinates, for the occlusion cull
    points  float32[3 * n]

## The pose convention, and how it is verified

TartanAir V2 poses are NED: body x forward, y right, z down. The image frame is
the usual computer-vision one: x right, y down, z forward. So a camera-frame
point maps to the body frame by a permutation, NOT by identity:

    body = (Zc, Xc, Yc)          forward, right, down

then world = R(quaternion) @ body + t.

Getting this wrong does not raise. It produces a plausible-looking cloud whose
frames do not agree with each other, and a single frame looks perfectly fine
because a single frame is self-consistent under any convention.

`--verify` therefore checks it the only way that works: fuse K consecutive
frames and watch how the occupied-block count grows. Frames of the same room
from nearby viewpoints mostly re-observe the same surfaces, so with the right
convention the block count grows sharply for the first few frames and then
flattens. With a wrong permutation each frame lands somewhere else and the
count grows nearly linearly. The ratio between the two behaviours is large and
needs no threshold tuning to read.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tartanair import CX, CY, FX, FY, HEIGHT, WIDTH, read_depth  # noqa: E402


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from (qx, qy, qz, qw)."""
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# Camera-frame axes expressed in body (NED) axes. The default is the correct
# one; the alternatives exist so --verify can demonstrate that it is correct
# rather than assert it.
CONVENTIONS = {
    "ned": (2, 0, 1),       # body = (Zc, Xc, Yc): forward, right, down
    "identity": (0, 1, 2),  # wrong on purpose: treats camera axes as body axes
    "swapped": (2, 1, 0),   # wrong on purpose
}


def unproject(depth: np.ndarray, pose: np.ndarray, stride: int, max_range: float,
              convention: str = "ned") -> tuple[np.ndarray, np.ndarray]:
    """One frame of depth to world points, plus the camera centre."""
    ys, xs = np.mgrid[0:HEIGHT:stride, 0:WIDTH:stride]
    z = depth[::stride, ::stride]
    ok = np.isfinite(z) & (z > 0.05) & (z < max_range)

    xs, ys, z = xs[ok], ys[ok], z[ok]
    # Camera frame: x right, y down, z forward.
    xc = (xs - CX) * z / FX
    yc = (ys - CY) * z / FY
    zc = z

    cam_pts = np.stack([xc, yc, zc], axis=1)
    perm = CONVENTIONS[convention]
    body = cam_pts[:, perm]

    t = pose[:3]
    R = quat_to_R(pose[3:])
    world = body @ R.T + t
    return world.astype(np.float32), t.astype(np.float32)


def write_bin(path: Path, pts: np.ndarray, cam: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(b"OSNP")
        f.write(struct.pack("<ii", 1, len(pts)))
        f.write(struct.pack("<fff", *cam))
        f.write(pts.reshape(-1).astype("<f4").tobytes())


def load_traj(traj: Path):
    poses = np.loadtxt(traj / "pose_lcam_front.txt")
    depths = sorted((traj / "depth_lcam_front").glob("*.png"))
    return poses, depths


def verify(traj: Path, k: int, stride: int, max_range: float, voxel: float) -> int:
    """Fuse k frames under each convention and report block-count growth."""
    poses, depths = load_traj(traj)
    k = min(k, len(depths), len(poses))
    print(f"verifying pose convention on {traj.name}, {k} frames, "
          f"stride {stride}, voxel {voxel} m\n")

    block = voxel * 8.0  # kBlockDim
    results = {}
    for name in CONVENTIONS:
        occupied: set[tuple[int, int, int]] = set()
        counts = []
        for i in range(k):
            d = read_depth(depths[i])
            pts, _ = unproject(d, poses[i], stride, max_range, name)
            b = np.floor(pts / block).astype(np.int64)
            occupied.update(map(tuple, b))
            counts.append(len(occupied))
        # Growth of the last frame relative to the first: with correct poses
        # later frames mostly re-observe known surfaces, so the marginal
        # contribution collapses.
        marginal_first = counts[0]
        marginal_last = counts[-1] - counts[-2] if k > 1 else counts[0]
        ratio = marginal_last / marginal_first if marginal_first else float("nan")
        results[name] = (counts[-1], ratio)
        print(f"  {name:<9} total blocks {counts[-1]:>8}   "
              f"last-frame marginal / first-frame {ratio:6.3f}")

    best = min(results, key=lambda n: results[n][1])
    print(f"\n  lowest marginal growth: {best}")
    if best == "ned":
        print("  PASS: the NED permutation re-observes surfaces; the others do not.")
        return 0
    print("  FAIL: expected 'ned' to win. Do not use this data until resolved.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traj", type=Path, help="e.g. .../RetroOffice/Data_easy/P000")
    ap.add_argument("--out", type=Path, help="output .bin (single frame)")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--warm-out", type=Path,
                    help="also write the K frames BEFORE --frame, concatenated, "
                         "for pre-filling a volume")
    ap.add_argument("--warm-frames", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-range", type=float, default=5.0)
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-frames", type=int, default=10)
    args = ap.parse_args()

    if args.verify:
        return verify(args.traj, args.verify_frames, max(args.stride, 4),
                      args.max_range, args.voxel)

    poses, depths = load_traj(args.traj)
    d = read_depth(depths[args.frame])
    pts, cam = unproject(d, poses[args.frame], args.stride, args.max_range)
    if args.out:
        write_bin(args.out, pts, cam)
        print(f"wrote {args.out}: {len(pts)} points, cam {cam}")

    if args.warm_out:
        lo = max(0, args.frame - args.warm_frames)
        acc = []
        for i in range(lo, args.frame):
            wd = read_depth(depths[i])
            wp, _ = unproject(wd, poses[i], args.stride, args.max_range)
            acc.append(wp)
        allp = np.concatenate(acc) if acc else np.zeros((0, 3), np.float32)
        # The warm cloud is integrated outside the timed window purely to put
        # the volume in a realistic state, so its camera centre is irrelevant
        # and the frame's own is reused.
        write_bin(args.warm_out, allp, cam)
        print(f"wrote {args.warm_out}: {len(allp)} points from frames {lo}..{args.frame - 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
