#!/usr/bin/env python3
"""Build the 1200x630 social preview card for the project page.

Without one, a link to this page has nothing for LinkedIn, Slack or Mastodon to
render, so they fall back to whatever other URL in the post does have metadata.
That is how a post about this paper ended up previewing a GitHub pull request.

The card is generated rather than drawn by hand for the same reason every other
figure here is: the headline ratios on it come from the sweep CSV, so they
cannot drift away from the paper they are advertising.
"""

import csv
import collections
import pathlib
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.image as mpimg          # noqa: E402

BG = "#fbfaf8"
INK = "#14171c"
MUTED = "#5a6472"
RULE = "#c4c8cf"
RUST = "#a63a0c"
TRITON = "#5f4390"

SANS = ["DejaVu Sans"]
MONO = ["DejaVu Sans Mono"]


def stage_range(csv_path, arm, stage):
    raw = collections.defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        raw[(r["device"], r["cell"], r["arm"], r["stage"])].append(float(r["p50_ms"]))
    med = {k: statistics.median(v) for k, v in raw.items()}
    cells = sorted({k[1] for k in med})
    devs = sorted({k[0] for k in med})
    vals = [med[(d, c, arm, stage)] / med[(d, c, "A3-cuda", stage)]
            for d in devs for c in cells
            if (d, c, arm, stage) in med and (d, c, "A3-cuda", stage) in med]
    return min(vals), max(vals)


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    sweep = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
        sorted((root / "results").glob("sweep_*/sweep.csv"))[-1]

    ra = stage_range(sweep, "A4-rust", "allocate")
    ta = stage_range(sweep, "A5-triton", "allocate")
    ru = stage_range(sweep, "A4-rust", "update")
    tu = stage_range(sweep, "A5-triton", "update")

    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(BG)

    # The render carries the "this is 3D reconstruction" signal instantly,
    # which is what makes a scroller stop; the numbers carry the claim.
    img_path = root / "docs/renders/lf-hi.lost-vs-triton.png"
    ax = fig.add_axes([0.615, 0.06, 0.36, 0.88])
    ax.imshow(mpimg.imread(str(img_path)))
    ax.axis("off")

    def txt(x, y, s, size, color=INK, weight="normal", family=SANS, style="normal"):
        fig.text(x, y, s, fontsize=size, color=color, weight=weight,
                 family=family, style=style, va="top", ha="left")

    txt(0.055, 0.90, "What Irregularity Costs", 38, INK, "bold")
    txt(0.055, 0.775, "CUDA C++, Rust, and Triton on a", 19, MUTED, style="italic")
    txt(0.055, 0.715, "hash-blocked GPU workload", 19, MUTED, style="italic")

    fig.add_artist(plt.Line2D([0.055, 0.55], [0.645, 0.645], color=RULE, lw=1.2))

    txt(0.055, 0.595, "REGULAR STAGE", 12, MUTED, family=MONO)
    txt(0.055, 0.535, f"Rust {ru[0]:.2f}–{ru[1]:.2f}×", 21, RUST, "bold", MONO)
    txt(0.30, 0.535, f"Triton {tu[0]:.1f}–{tu[1]:.1f}×", 21, TRITON, "bold", MONO)

    txt(0.055, 0.415, "IRREGULAR STAGE", 12, MUTED, family=MONO)
    txt(0.055, 0.355, f"Rust {ra[0]:.2f}–{ra[1]:.2f}×", 21, RUST, "bold", MONO)
    txt(0.30, 0.355, f"Triton {ta[0]:.1f}–{ta[1]:.1f}×", 21, TRITON, "bold", MONO)

    txt(0.055, 0.225, "Language choice is nearly free on the work that is", 15, INK)
    txt(0.055, 0.168, "usually benchmarked, and expensive on the work", 15, INK)
    txt(0.055, 0.111, "that is not.", 15, INK)

    txt(0.055, 0.045, "arXiv:2608.08287", 13, MUTED, family=MONO)

    out = root / "docs/og-image.png"
    fig.savefig(out, facecolor=BG, dpi=100)
    plt.close(fig)
    w, h = 1200, 630
    print(f"  {out.relative_to(root)}  {out.stat().st_size/1024:.0f} KiB  target {w}x{h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
