#!/usr/bin/env python3
"""Render the fused point clouds to images, for the paper and the page.

These read the same reduced assets the interactive viewer loads, so a reader
looking at the printed figure and a reader dragging the canvas are looking at
the same points. Rendering from the raw dumps instead would let the two drift.

The rasteriser is written out here rather than pulled from a library because
what it has to do is small and specific: project points, splat each as a disc
of a few pixels, keep the nearest by depth, and shade by the surface normal
the TSDF gave us. Matplotlib's 3D scatter has no depth buffer and paints in
draw order, which for a closed surface means the back of the object shows
through the front.
"""

import json
import math
import pathlib
import struct
import sys

import numpy as np

# Neutral ink with a faint warm bias: the geometry is not categorical data, so
# it does not take a series hue. Reads the same on paper and on screen.
KEY = np.array([0.96, 0.93, 0.90])
FILL = np.array([0.42, 0.47, 0.58])
AMBIENT = np.array([0.13, 0.13, 0.15])


def read_bin(path):
    b = path.read_bytes()
    magic, _ver, n = struct.unpack_from("<III", b, 0)
    if magic != 0x32475350:
        raise ValueError(f"{path}: bad magic")
    lo = np.array(struct.unpack_from("<fff", b, 12))
    hi = np.array(struct.unpack_from("<fff", b, 24))
    q = np.frombuffer(b, dtype="<u2", count=n * 3, offset=36).reshape(n, 3)
    o = np.frombuffer(b, dtype=np.int8, count=n * 2, offset=36 + n * 6).reshape(n, 2)
    pos = lo + q.astype(np.float64) / 65535.0 * (hi - lo)
    e = o.astype(np.float64) / 127.0
    nz = 1.0 - np.abs(e[:, 0]) - np.abs(e[:, 1])
    nx, ny = e[:, 0].copy(), e[:, 1].copy()
    low = nz < 0
    if low.any():
        nx[low], ny[low] = ((1 - np.abs(e[low, 1])) * np.sign(e[low, 0]),
                            (1 - np.abs(e[low, 0])) * np.sign(e[low, 1]))
    nor = np.stack([nx, ny, nz], axis=1)
    nor /= np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-9)
    return pos, nor


def look_at(yaw, pitch, dist, center):
    eye = center + dist * np.array([
        math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)])
    z = eye - center
    z /= np.linalg.norm(z)
    x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    nx = np.linalg.norm(x)
    x = np.array([1.0, 0.0, 0.0]) if nx < 1e-6 else x / nx
    y = np.cross(z, x)
    return np.stack([x, y, z]), eye


def render(pos, nor, size=900, yaw=0.9, pitch=0.42, radius=None, supersample=2):
    """Orthographic z-buffered point splat. Returns float RGB and coverage."""
    # Splat radius tracks point spacing, which goes as 1/sqrt(n): a fixed
    # radius either leaves a dense cloud speckled or turns a sparse one into
    # a blob. Both extremes are in this gallery -- 2k points and 170k.
    if radius is None:
        radius = int(min(8, max(2, round(1.3 * size / math.sqrt(max(len(pos), 1))))))
    S = size * supersample
    center = 0.5 * (pos.min(axis=0) + pos.max(axis=0))
    scale = float(np.abs(pos - center).max()) or 1.0
    R, eye = look_at(yaw, pitch, scale * 4.0, center)

    cam = (pos - eye) @ R.T
    ncam = nor @ R.T
    # Orthographic keeps the sphere a circle and the plane a rectangle, which
    # is what these scenes are; perspective would only add foreshortening the
    # reader has to discount.
    k = S / (2.35 * scale)
    px = (cam[:, 0] * k + S / 2).astype(np.int32)
    py = (-cam[:, 1] * k + S / 2).astype(np.int32)
    depth = -cam[:, 2]

    lam_key = np.clip(ncam @ np.array([0.35, 0.62, 0.70]) / 1.0, 0, 1)
    lam_fill = np.clip(ncam @ np.array([-0.55, -0.25, 0.35]), 0, 1)
    shade = AMBIENT + KEY * lam_key[:, None] * 0.86 + FILL * lam_fill[:, None] * 0.32

    zbuf = np.full(S * S, np.inf)
    col = np.zeros((S * S, 3))
    r = radius * supersample
    offsets = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
               if dx * dx + dy * dy <= r * r]
    # Painting far-to-near and testing depth per offset keeps the nearest
    # surface; the sort makes the result independent of input order.
    order = np.argsort(-depth)
    px, py, depth, shade = px[order], py[order], depth[order], shade[order]
    for dx, dy in offsets:
        x, y = px + dx, py + dy
        ok = (x >= 0) & (x < S) & (y >= 0) & (y < S)
        idx = y[ok] * S + x[ok]
        d = depth[ok]
        keep = d < zbuf[idx]
        # np.minimum.at resolves ties within one offset pass deterministically.
        sel = np.flatnonzero(keep)
        zbuf[idx[sel]] = d[sel]
        col[idx[sel]] = shade[ok][sel]

    img = col.reshape(S, S, 3)
    cov = np.isfinite(zbuf).reshape(S, S).astype(np.float64)
    if supersample > 1:
        img = img.reshape(size, supersample, size, supersample, 3).mean(axis=(1, 3))
        cov = cov.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    return img, cov


def save_png(path, img, cov):
    """Straight-alpha PNG so the page's background shows through in both themes."""
    import zlib
    h, w = cov.shape
    a = np.clip(cov, 0, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = np.where(a[..., None] > 0, img / np.maximum(a[..., None], 1e-6), 0.0)
    rgba = np.concatenate([np.clip(rgb, 0, 1), a[..., None]], axis=2)
    data = (rgba * 255 + 0.5).astype(np.uint8)
    raw = b"".join(b"\x00" + data[y].tobytes() for y in range(h))

    def chunk(tag, payload):
        c = struct.pack(">I", len(payload)) + tag + payload
        return c + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    path.write_bytes(png)
    return len(png)


# The rust hue from the validated chart palette, used only to mark loss.
LOSS = np.array([0.65, 0.23, 0.05])


def render_diff(pos_a, nor_a, pos_b, size, yaw, pitch, leaf):
    """Render cloud A, marking the parts of it that cloud B does not have.

    Two renders side by side cannot show a one-percent difference; the eye has
    no way to find it. Marking the missing points on the surface that does have
    them shows both how much was lost and, more usefully, *where* -- whether
    the loss is scattered evenly or concentrated, which is the difference
    between rounding and a structural failure.
    """
    def keys(p):
        k = np.floor(p / leaf).astype(np.int64)
        return ((k[:, 0] + (1 << 20)) << 42) | ((k[:, 1] + (1 << 20)) << 21) | \
               (k[:, 2] + (1 << 20))

    missing = ~np.isin(keys(pos_a), keys(pos_b))
    img, cov = render(pos_a, nor_a, size=size, yaw=yaw, pitch=pitch)
    if missing.any():
        # Draw the lost points alone, then composite them over the surface.
        hi, hcov = render(pos_a[missing], nor_a[missing], size=size, yaw=yaw,
                          pitch=pitch, radius=4)
        tint = LOSS * np.maximum(hi.sum(axis=2, keepdims=True) / 1.6, 0.55)
        m = hcov[..., None]
        img = img * (1 - m) + tint * m
        cov = np.maximum(cov, hcov)
    return img, cov, int(missing.sum())


# The cells worth a picture: one per distinct thing the sweep varies, rather
# than all fifteen, most of which are the same sphere at a different density.
GALLERY = [
    ("tartan-warm", "real depth, warm volume", 0.75, 0.30),
    ("base", "sphere, 320k points", 0.9, 0.42),
    ("pts-20k", "sphere, 20k points", 0.9, 0.42),
    ("plane-320k", "plane, 320k points", -0.5, 0.40),
    ("r-2.0", "sphere, 2 m radius", 0.9, 0.42),
    ("lf-hi", "load factor 1.05", 0.9, 0.42),
]


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    geo = root / "docs/geometry"
    out_web = root / "docs/renders"
    out_paper = root / "paper/figures"
    out_web.mkdir(parents=True, exist_ok=True)
    out_paper.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((geo / "manifest.json").read_text())
    entries = {e["cell"]: e for e in manifest["cells"]}

    size = int(sys.argv[1]) if len(sys.argv) > 1 else 760
    made = []
    for cell, label, yaw, pitch in GALLERY:
        e = entries.get(cell)
        if not e:
            print(f"  {cell}: not exported, skipping")
            continue
        base_arm = "A3-cuda" if "A3-cuda" in e["files"] else sorted(e["files"])[0]
        pos, nor = read_bin(geo / e["files"][base_arm]["file"])
        img, cov = render(pos, nor, size=size, yaw=yaw, pitch=pitch)
        name = f"{cell}.png"
        n = save_png(out_web / name, img, cov)
        save_png(out_paper / name, img, cov)
        made.append({"cell": cell, "label": label, "file": name,
                     "points": e["files"][base_arm]["points"], "lost": 0})
        print(f"  {cell:12s} {base_arm:10s} {e['files'][base_arm]['points']:>7,} pts"
              f" -> {name} ({n/1024:.0f} KiB)")

        for arm in sorted(a for a in e["files"] if a != base_arm):
            other, _ = read_bin(geo / e["files"][arm]["file"])
            dimg, dcov, lost = render_diff(pos, nor, other, size, yaw, pitch,
                                           e["files"][base_arm]["leaf_m"])
            dname = f"{cell}.lost-vs-{arm.split('-')[1]}.png"
            n = save_png(out_web / dname, dimg, dcov)
            save_png(out_paper / dname, dimg, dcov)
            made.append({"cell": cell, "label": label + ", what " + arm.split("-")[1] + " lost",
                         "file": dname, "points": e["files"][arm]["points"], "lost": lost})
            print(f"  {cell:12s} vs {arm:8s} {lost:>7,} points absent"
                  f" -> {dname} ({n/1024:.0f} KiB)")

    (out_web / "gallery.json").write_text(json.dumps(made, indent=1))
    print(f"\n  {len(made)} renders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
