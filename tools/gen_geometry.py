#!/usr/bin/env python3
"""Turn the harness's raw surface dumps into assets the project page can load.

`bench_arms --export-geometry` writes the mesh exactly as extraction produced
it: interleaved position/normal floats, up to 261 MiB for the real-data cell.
That is the right thing to leave the GPU with and the wrong thing to put on a
web page, so the reduction happens here, in one place, where it can be read.

The reduction is a voxel-grid average, not a triangle decimation. Dropping
triangles from a marching-tets surface leaves holes that look like
reconstruction failures rather than like compression, which would misrepresent
the result. Averaging vertices into a grid keeps density uniform and is also
the honest picture of what a TSDF actually is: the output is voxel-derived, so
showing it as one lit point per occupied cell is closer to the underlying data
than a smoothed mesh would be.

Positions quantise to uint16 across the cell's own bounding box, and normals
octahedron-encode to two int8, so a point costs 8 bytes.
"""

import json
import pathlib
import struct
import sys

import numpy as np

MAGIC_IN = 0x314E534F   # "OSN1", written by write_geometry in bench_arms.cu
MAGIC_OUT = 0x32475350  # "PSG2"

# Enough points to read the shape at a glance without making the page wait.
# The real-data room gets more because it is the one with detail worth seeing.
BUDGET = {"tartan-cold": 220_000, "tartan-warm": 220_000}
BUDGET_DEFAULT = 120_000


def read_raw(path):
    buf = path.read_bytes()
    magic, n = struct.unpack_from("<II", buf, 0)
    if magic != MAGIC_IN:
        raise ValueError(f"{path}: bad magic {magic:#x}")
    a = np.frombuffer(buf, dtype=np.float32, count=n * 6, offset=8).reshape(n, 6)
    return a[:, 0:3].astype(np.float64), a[:, 3:6].astype(np.float64)


def voxel_average(pos, nor, leaf):
    """Average every vertex falling in the same cube of side `leaf`."""
    keys = np.floor(pos / leaf).astype(np.int64)
    # A single sortable key per cell; the offset keeps negatives well-ordered.
    k = ((keys[:, 0] + (1 << 20)) << 42) | ((keys[:, 1] + (1 << 20)) << 21) | \
        (keys[:, 2] + (1 << 20))
    order = np.argsort(k, kind="stable")
    k = k[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    counts = np.diff(np.r_[starts, k.size]).astype(np.float64)

    def group_mean(v):
        cs = np.cumsum(v[order], axis=0)
        tot = cs[np.r_[starts[1:], k.size] - 1]
        tot[1:] -= cs[starts[1:] - 1]
        return tot / counts[:, None]

    return group_mean(pos), group_mean(nor)


def reduce_to_budget(pos, nor, budget, extent):
    """Grow the grid until the point count fits, then average once more."""
    leaf = max(extent / 512.0, 1e-6)
    for _ in range(24):
        n_cells = np.unique(np.floor(pos / leaf).astype(np.int64), axis=0).shape[0]
        if n_cells <= budget:
            break
        leaf *= 1.26  # ~2x in cell count per step
    return (*voxel_average(pos, nor, leaf), leaf)


def oct_encode(n):
    """Unit normals to two signed bytes, via the octahedron mapping."""
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-12)
    s = np.abs(n).sum(axis=1, keepdims=True)
    s[s < 1e-12] = 1.0
    p = n[:, :2] / s
    lower = n[:, 2] < 0
    if lower.any():
        wrapped = (1.0 - np.abs(p[lower][:, ::-1])) * \
            np.where(p[lower] >= 0, 1.0, -1.0)
        p[lower] = wrapped
    return np.clip(np.round(p * 127.0), -127, 127).astype(np.int8)


def write_bin(path, pos, nor, bbox_min, bbox_max):
    span = np.maximum(bbox_max - bbox_min, 1e-9)
    q = np.clip(np.round((pos - bbox_min) / span * 65535.0), 0, 65535).astype("<u2")
    o = oct_encode(nor)
    head = struct.pack("<IIIffffff", MAGIC_OUT, 2, len(pos),
                       *bbox_min.astype(np.float32), *bbox_max.astype(np.float32))
    path.write_bytes(head + q.tobytes() + o.tobytes())
    return len(head) + q.nbytes + o.nbytes


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/geo")
    out = root / "docs/geometry"
    out.mkdir(parents=True, exist_ok=True)

    raws = sorted(src.glob("*.raw"))
    by_cell = {}
    for r in raws:
        cell, arm, _ = r.name.rsplit(".", 2)
        by_cell.setdefault(cell, {})[arm] = r
    if not by_cell:
        print(f"no .raw files in {src}")
        return 1

    entries = []
    for cell in sorted(by_cell):
        arms = by_cell[cell]
        counts = {}
        for arm, path in sorted(arms.items()):
            with path.open("rb") as f:
                counts[arm] = struct.unpack("<II", f.read(8))[1]

        # Ship A3's surface. Where an arm genuinely built something different
        # rather than merely rounding differently, ship that one too: the
        # difference is a result, not an artefact to hide.
        ship = {"A3-cuda"}
        base = counts.get("A3-cuda", 0)
        for arm, n in counts.items():
            if base and abs(n - base) / base > 1e-3:
                ship.add(arm)

        files = {}
        for arm in sorted(ship):
            pos, nor = read_raw(arms[arm])
            lo, hi = pos.min(axis=0), pos.max(axis=0)
            budget = BUDGET.get(cell, BUDGET_DEFAULT)
            p, nn, leaf = reduce_to_budget(pos, nor, budget, float((hi - lo).max()))
            name = f"{cell}.{arm}.bin"
            size = write_bin(out / name, p, nn, lo, hi)
            files[arm] = {"file": name, "points": int(len(p)),
                          "bytes": size, "leaf_m": round(leaf, 5)}
            print(f"  {cell:12s} {arm:10s} {counts[arm]:>9,} verts -> "
                  f"{len(p):>7,} pts  {size/1024:6.0f} KiB")

        entries.append({
            "cell": cell,
            "verts": counts,
            "agree": len({round(v / max(base, 1), 3) for v in counts.values()}) == 1,
            "files": files,
        })

    (out / "manifest.json").write_text(json.dumps({"cells": entries}, indent=1))
    total = sum(f["bytes"] for e in entries for f in e["files"].values())
    print(f"\n  {len(entries)} cells, {total/1048576:.1f} MiB total "
          f"(loaded one cell at a time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
