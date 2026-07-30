"""TartanAir V2 loader and format verifier.

The V1 documentation is wrong about V2 in ways that corrupt results silently
rather than crash. Every constant here was measured from the data on
2026-07-29, not taken from the docs. See docs/DATASET.md for the full list.

The two traps that motivated this file:

  1. Resolution is 640x640, not the documented 640x480, so cy = 320 not 240.
     Using the documented value biases every backprojected point vertically.

  2. Depth is float32 bit-packed into a 4-channel PNG. It MUST be read with
     cv2.imread(..., IMREAD_UNCHANGED).view("<f4"). Reading with PIL permutes
     the four bytes of each float (PIL reorders to RGBA, the on-disk layout is
     BGRA) and yields values in a *plausible* 0.13-8 m range with the structure
     destroyed. A range assertion passes it. verify() catches it via a gradient
     check, which is why that check exists.

Usage:
    python tools/tartanair.py <traj_dir>        # verify a trajectory
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# --- Measured format constants. Do not "correct" these from the docs. -------
WIDTH = 640
HEIGHT = 640          # docs say 480
FX = FY = 320.0
CX = CY = 320.0       # docs say cy = 240
BASELINE_M = 0.25     # verified 0.250000011 +/- 1.3e-7 over 129 frames
FX_TIMES_BASELINE = FX * BASELINE_M   # 80.0, so Z = 80/disparity

# Valid download modalities. NOTE: "pose" is NOT one; poses ship with each
# trajectory automatically.
MODALITIES = ("image", "depth", "seg", "imu", "lidar", "flow", "events", "mp4")


@dataclass
class Frame:
    """One rectified stereo frame with ground-truth depth."""
    left: np.ndarray        # (H, W, 3) uint8, BGR as returned by cv2
    right: np.ndarray       # (H, W, 3) uint8, BGR
    depth_left: np.ndarray  # (H, W) float32, planar Z in metres
    pose_left: np.ndarray   # (7,) tx ty tz qx qy qz qw, NED
    pose_right: np.ndarray  # (7,)
    index: int


def read_depth(path: str | Path) -> np.ndarray:
    """Decode a TartanAir V2 depth PNG to float32 metres (planar Z).

    Uses cv2 specifically: it preserves the on-disk BGRA byte order that the
    float32 view depends on. Do not substitute PIL.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"could not read depth image: {path}")
    if raw.ndim != 3 or raw.shape[2] != 4:
        raise ValueError(
            f"expected 4-channel depth PNG, got shape {raw.shape} for {path}"
        )
    return raw.view("<f4").squeeze()


def depth_to_disparity(depth_m: np.ndarray) -> np.ndarray:
    """Planar Z in metres to disparity in pixels.

    Uses the raw stored depth deliberately. Do NOT apply the tartanair
    package's depth_to_dist() first: disparity is defined against planar Z,
    and the ray-length conversion injects a radially varying error that looks
    like lens distortion.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return FX_TIMES_BASELINE / depth_m


def depth_error_per_pixel_disparity(z_m: float) -> float:
    """Depth error in metres induced by 1 px of disparity error at range z.

    dZ = Z^2 / (fx * B). Quadratic, so 1 px costs 1.25 cm at 1 m but 31 cm at
    5 m. This is what bounds useful voxel size and fusion range.
    """
    return z_m * z_m / FX_TIMES_BASELINE


def max_useful_range(voxel_m: float) -> float:
    """Range beyond which 1 px of disparity error exceeds one voxel."""
    return float(np.sqrt(voxel_m * FX_TIMES_BASELINE))


class Trajectory:
    """A single TartanAir V2 trajectory, e.g. <env>/Data_easy/P000."""

    def __init__(self, root: str | Path, cam: str = "front"):
        self.root = Path(root)
        self.cam = cam
        self.left_dir = self.root / f"image_lcam_{cam}"
        self.right_dir = self.root / f"image_rcam_{cam}"
        self.depth_l_dir = self.root / f"depth_lcam_{cam}"
        self.depth_r_dir = self.root / f"depth_rcam_{cam}"
        for d in (self.left_dir, self.right_dir, self.depth_l_dir):
            if not d.is_dir():
                raise FileNotFoundError(f"missing {d}")
        self.pose_left = np.loadtxt(self.root / f"pose_lcam_{cam}.txt")
        self.pose_right = np.loadtxt(self.root / f"pose_rcam_{cam}.txt")
        self.n = len(sorted(self.left_dir.glob("*.png")))

    def __len__(self) -> int:
        return self.n

    def frame(self, i: int) -> Frame:
        left = cv2.imread(str(self.left_dir / f"{i:06d}_lcam_{self.cam}.png"))
        right = cv2.imread(str(self.right_dir / f"{i:06d}_rcam_{self.cam}.png"))
        depth = read_depth(self.depth_l_dir / f"{i:06d}_lcam_{self.cam}_depth.png")
        return Frame(left, right, depth, self.pose_left[i], self.pose_right[i], i)

    # -- verification ---------------------------------------------------

    def verify(self, verbose: bool = True) -> bool:
        """Assert the measured format constants actually hold for this data.

        Run this on any new environment before trusting it. It has already
        caught one real decode bug; the gradient check is the part that
        distinguishes correct depth from byte-permuted depth, because the
        value range alone does not.
        """
        ok = True

        def check(label: str, passed: bool, detail: str = "") -> None:
            nonlocal ok
            ok &= passed
            if verbose:
                print(f"  [{'PASS' if passed else 'FAIL'}] {label}"
                      + (f"  {detail}" if detail else ""))

        f = self.frame(0)

        check("resolution 640x640",
              f.left.shape[:2] == (HEIGHT, WIDTH), f"got {f.left.shape[:2]}")
        check("depth is float32",
              f.depth_left.dtype == np.float32, str(f.depth_left.dtype))
        check("depth finite and positive",
              bool(np.isfinite(f.depth_left).all() and (f.depth_left > 0).all()))

        # Byte-permuted depth is high-frequency noise; real depth is piecewise
        # smooth. Median absolute gradient separates them by orders of
        # magnitude, where the value range alone does not.
        grad = np.abs(np.diff(f.depth_left, axis=1))
        med_grad = float(np.median(grad))
        check("depth is smooth (not byte-permuted)",
              med_grad < 0.05, f"median |dZ/dx| = {med_grad:.4f} m")

        # Rectification: pure horizontal translation, no relative rotation.
        from scipy.spatial.transform import Rotation
        ned_r_cam = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        angs, tvecs = [], []
        for a, b in zip(self.pose_left, self.pose_right):
            rl = Rotation.from_quat(a[3:]).as_matrix() @ ned_r_cam
            rr = Rotation.from_quat(b[3:]).as_matrix() @ ned_r_cam
            rrel = rl.T @ rr
            angs.append(np.degrees(np.arccos(np.clip((np.trace(rrel) - 1) / 2, -1, 1))))
            tvecs.append(rl.T @ (b[:3] - a[:3]))
        angs = np.asarray(angs)
        t = np.asarray(tvecs)

        check("relative rotation ~ 0",
              angs.max() < 1e-4, f"max {angs.max():.2e} deg")
        check("baseline 0.25 m on camera x",
              abs(t[:, 0].mean() - BASELINE_M) < 1e-6, f"{t[:, 0].mean():.9f} m")
        check("no vertical/forward offset",
              max(abs(t[:, 1]).max(), abs(t[:, 2]).max()) < 1e-5,
              f"max {max(abs(t[:, 1]).max(), abs(t[:, 2]).max()):.2e} m")

        return ok

    def disparity_stats(self, stride: int = 8) -> dict:
        """Disparity distribution, for choosing the max-disparity mask.

        Matters because a stereo model has a bounded search range: GT
        disparities above that bound are structurally unpredictable, so those
        pixels are guaranteed maximal errors and must be masked and reported
        rather than silently included.
        """
        zs = [self.frame(i).depth_left.ravel() for i in range(0, self.n, stride)]
        z = np.concatenate(zs)
        d = depth_to_disparity(z)
        return {
            "frames_sampled": len(zs),
            "megapixels": z.size / 1e6,
            "depth_p1": float(np.percentile(z, 1)),
            "depth_p50": float(np.percentile(z, 50)),
            "depth_p99": float(np.percentile(z, 99)),
            "disp_p50": float(np.percentile(d, 50)),
            "disp_p99": float(np.percentile(d, 99)),
            "disp_max": float(d.max()),
            "frac_over_192": float((d > 192).mean()),
            "frac_over_256": float((d > 256).mean()),
        }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    traj = Trajectory(argv[1])
    print(f"trajectory: {traj.root}  ({len(traj)} frames)\n")
    print("format verification:")
    ok = traj.verify()
    print("\ndisparity distribution:")
    for k, v in traj.disparity_stats().items():
        print(f"  {k:16s} {v:.4f}" if isinstance(v, float) else f"  {k:16s} {v}")
    print(f"\n=> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
