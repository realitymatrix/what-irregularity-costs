"""Cross-validate crates/mesh-metrics against Open3D.

The gate blocks every arm, so its metric must be checked against an independent
implementation rather than only against itself. Run after `verify_metrics`,
which writes the PLYs and its own numbers.

Open3D's RaycastingScene.compute_distance is the point-to-surface query, which
is what mesh-metrics computes. Note that `compute_point_cloud_distance` is NOT
the right comparison: it is point-to-point and carries a sample-spacing floor.

Usage: python tools/verify_metrics_open3d.py <artifacts_dir>
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import open3d as o3d


def surface_distance(src: o3d.geometry.TriangleMesh,
                     dst: o3d.geometry.TriangleMesh,
                     n: int, seed: int) -> np.ndarray:
    """Distances from points sampled on `src` to the surface of `dst`."""
    o3d.utility.random.seed(seed)
    pts = np.asarray(src.sample_points_uniformly(n).points, dtype=np.float32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(dst))
    return scene.compute_distance(o3d.core.Tensor(pts)).numpy()


def main(argv: list[str]) -> int:
    d = pathlib.Path(argv[1] if len(argv) > 1 else "artifacts/metrics")
    rust = json.loads((d / "rust_metrics.json").read_text())

    a = o3d.io.read_triangle_mesh(str(d / "sphere_a.ply"))
    b = o3d.io.read_triangle_mesh(str(d / "sphere_b.ply"))
    n = int(rust["n_samples"])

    ab = surface_distance(a, b, n, 1)
    ba = surface_distance(b, a, n, 2)

    o3d_mean = 0.5 * (ab.mean() + ba.mean())
    o3d_haus = max(ab.max(), ba.max())
    o3d_p99 = max(np.percentile(ab, 99), np.percentile(ba, 99))

    rows = [
        ("mean surface", rust["mean_surface"], o3d_mean, 1e-3),
        ("hausdorff", rust["hausdorff"], o3d_haus, 5e-3),
        ("hausdorff p99", rust["hausdorff_p99"], o3d_p99, 2e-3),
    ]
    print("=== mesh-metrics vs Open3D (concentric spheres, dr = 0.05) ===")
    print(f"{'metric':16s} {'rust':>12s} {'open3d':>12s} {'|diff|':>10s}  tol")
    ok = True
    for name, r, o, tol in rows:
        diff = abs(r - o)
        good = diff < tol
        ok &= good
        print(f"{name:16s} {r:12.6f} {o:12.6f} {diff:10.2e}  {tol:.0e}  "
              f"[{'PASS' if good else 'FAIL'}]")

    # Both must also recover the analytic answer, which is exactly dr = 0.05.
    for label, val in (("rust", rust["mean_surface"]), ("open3d", o3d_mean)):
        good = abs(val - 0.05) < 2e-3
        ok &= good
        print(f"{label:16s} vs analytic 0.05: {val:.6f}  "
              f"[{'PASS' if good else 'FAIL'}]")

    print(f"\n=== {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
