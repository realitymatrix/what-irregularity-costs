#!/usr/bin/env python3
"""Generate the paper's figures and the project page's charts from the CSVs.

Every figure here is drawn from results/, never from hand-entered numbers, for
the same reason the tables are: a figure that is typed in by hand can disagree
with the data it claims to plot, and nothing in the build would notice.

Two output formats come off the same code path:

  * PDF into paper/figures/, included by main.tex.
  * SVG into docs/figures/, inlined into the project page by gen_page.py. The
    SVG is post-processed to swap fixed colours for CSS custom properties, so
    the charts follow the reader's light/dark theme instead of pinning one.

On colour: CUDA C++ is never a coloured series. It is the denominator of every
ratio plotted, so it is drawn as a neutral reference rule at 1.0 and only Rust
and Triton carry hues. That is the honest encoding, and it also sidesteps a
real accessibility failure -- a green/orange pair separates by only dE 1.6
under deuteranopia, which is to say not at all.
"""

import collections
import csv
import pathlib
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Validated with the palette checker in both modes: worst adjacent pair is
# dE 20.2 (light) and dE 23.8 (dark) under simulated CVD, against a dE 8 target.
LIGHT = {
    "rust": "#a63a0c", "triton": "#5f4390",
    "fg": "#14171c", "muted": "#5a6472", "rule": "#c4c8cf", "bg": "#fbfaf8",
}
DARK = {
    "rust": "#d1663a", "triton": "#9b7ae0",
    "fg": "#e4e7ea", "muted": "#98a1ac", "rule": "#3a414a", "bg": "#14161a",
}
# Sentinels let the SVG carry CSS variables that matplotlib would never emit.
VARS = {
    "rust": "var(--c-rust)", "triton": "var(--c-triton)",
    "fg": "var(--fg)", "muted": "var(--muted)", "rule": "var(--rule-hard)",
    "bg": "none",
}

ARM = {"A3-cuda": "cuda", "A4-rust": "rust", "A5-triton": "triton"}
REF_DEVICE = "NVIDIA GeForce RTX 5070 Ti"


def load(csv_path):
    """Collapse the sweep's repeated passes to a median per measured point."""
    raw = collections.defaultdict(list)
    meta = {}
    for r in csv.DictReader(open(csv_path)):
        arm = ARM.get(r["arm"])
        if arm is None:
            continue
        if r["axis"] == "coldwarm":
            continue
        key = (r["device_name"], r["cell"], arm, r["stage"])
        raw[key].append(float(r["p50_ms"]))
        meta[r["cell"]] = (r["axis"], int(r["points"]), int(r["blocks"]),
                           float(r["load_factor"]))
    return {k: statistics.median(v) for k, v in raw.items()}, meta


def ratio(med, device, cell, arm, stage):
    base = med.get((device, cell, "cuda", stage))
    got = med.get((device, cell, arm, stage))
    if not base or not got:
        return None
    return got / base


def style(ax, pal):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(pal["rule"])
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(which="both", colors=pal["muted"], labelsize=8,
                   length=3, width=0.8)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color(pal["muted"])


def fig_stages(med, meta, pal):
    """The thesis: the same two arms, priced on the regular and irregular stage.

    A range rather than a single number, because the spread across workloads is
    itself the finding -- Rust's irregular cost is workload-dependent in a way
    its regular cost is not.
    """
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    devices = sorted({k[0] for k in med})
    cells = sorted(meta)
    rows = [("triton", "allocate"), ("triton", "update"),
            ("rust", "allocate"), ("rust", "update")]
    labels = ["Triton, irregular", "Triton, regular",
              "Rust, irregular", "Rust, regular"]

    fmt = lambda v: f"{v:.2f}" if v < 10 else f"{v:.1f}"
    for i, (arm, stage) in enumerate(rows):
        vals = [v for v in (ratio(med, d, c, arm, stage)
                            for d in devices for c in cells) if v]
        lo, hi, mid = min(vals), max(vals), statistics.median(vals)
        ax.plot([lo, hi], [i, i], color=pal[arm], lw=3, solid_capstyle="round",
                zorder=2, alpha=.85)
        ax.plot([mid], [i], "o", color=pal[arm], ms=8, zorder=3,
                markeredgecolor=pal["bg"], markeredgewidth=1.5)
        ax.annotate(f"{fmt(lo)}–{fmt(hi)}×", (hi, i), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=pal["muted"])

    ax.axvline(1.0, color=pal["rule"], lw=1, zorder=1)
    ax.annotate("CUDA C++", (1.0, 3.62), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=pal["muted"], va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 60)
    ax.set_xticks([1, 2, 5, 10, 20, 40])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_ylim(-0.6, 3.9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"time relative to CUDA C++ (log scale), range over "
                  f"{len(cells)} workloads on {len(devices)} GPUs",
                  fontsize=8, color=pal["muted"])
    style(ax, pal)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig


def fig_axes(med, meta, pal):
    """The mechanism: two levers that move contention in opposite directions.

    If the cost were about problem size, both panels would slope the same way.
    They do not, which is what makes contention rather than size the
    explanation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    panels = [
        ("points", "more points, fixed geometry", "points (thousands)",
         lambda c: meta[c][1] / 1000.0, lambda v: f"{v:g}"),
        ("extent", "same points, spread wider", "table load factor",
         lambda c: meta[c][3] * 100.0, lambda v: f"{v:.1f}%"),
    ]
    for ax, (axis, title, xlabel, xof, xfmt) in zip(axes, panels):
        cells = sorted((c for c in meta if meta[c][0] == axis), key=xof)
        # The baseline cell varies point count, so it belongs on that panel;
        # on the extent panel r-1.0 already occupies the same position.
        if axis == "points" and "base" in meta:
            cells = sorted(cells + ["base"], key=xof)
        xs = [xof(c) for c in cells]
        for arm, name in (("triton", "Triton"), ("rust", "Rust")):
            ys = [ratio(med, REF_DEVICE, c, arm, "allocate") for c in cells]
            ax.plot(xs, ys, "-o", color=pal[arm], lw=2, ms=6, label=name,
                    markeredgecolor=pal["bg"], markeredgewidth=1.2, zorder=3)
            if axis == "extent":
                ax.annotate(name, (xs[-1], ys[-1]), xytext=(9, 0),
                            textcoords="offset points", va="center",
                            fontsize=8, color=pal["muted"])
        ax.axhline(1.0, color=pal["rule"], lw=1, zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(0.8, 40)
        ax.set_yticks([1, 2, 5, 10, 20])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.set_xscale("log")
        ax.set_xticks(xs, minor=False)
        ax.set_xticks([], minor=True)
        ax.set_xticklabels([xfmt(x) for x in xs])
        ax.set_title(title, fontsize=9, color=pal["fg"], pad=8)
        ax.set_xlabel(xlabel, fontsize=8, color=pal["muted"])
        style(ax, pal)

    axes[0].set_ylabel("allocate, relative to CUDA C++", fontsize=8,
                       color=pal["muted"])
    fig.tight_layout()
    return fig


def fig_floor(med, meta, pal):
    """The cost floor: hold the point count, vary the work, watch who moves.

    This is the sharpest form of the expressiveness claim. Every cell here
    launches the same number of threads over the same number of points; only
    the amount of insertion those threads have to do changes. An arm that can
    exit a probe early gets cheaper as the work falls. An arm that must run to
    a compile-time bound cannot.
    """
    cells = [c for c in meta if 319_000 <= meta[c][1] <= 321_000]
    cells.sort(key=lambda c: meta[c][2])
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    xs = [meta[c][2] for c in cells]

    series = [("cuda", "CUDA C++", pal["fg"]), ("rust", "Rust", pal["rust"]),
              ("triton", "Triton", pal["triton"])]
    for arm, name, colour in series:
        ys = [med[(REF_DEVICE, c, arm, "allocate")] for c in cells]
        ax.plot(xs, ys, "-o", color=colour, lw=2, ms=6, zorder=3,
                markeredgecolor=pal["bg"], markeredgewidth=1.2)
        span = max(ys) / min(ys)
        dy = {"cuda": -7, "rust": 7, "triton": 0}[arm]
        ax.annotate(f"{name}  ×{span:.2f}", (xs[-1], ys[-1]), xytext=(9, dy),
                    textcoords="offset points", va="center", fontsize=8,
                    color=colour if arm != "cuda" else pal["muted"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    # 1,060 and 1,160 are a hair apart on a log axis and their labels collide;
    # label the ends and the decades, and let the markers show the rest.
    shown = [xs[0], xs[-1]] + [x for x in xs if x not in (xs[0], xs[-1])
                               and min(abs(x - o) / o for o in (xs[0], xs[-1])) > 0.4]
    keep, last = [], 0
    for x in sorted(set(shown)):
        if last == 0 or x / last > 1.6:
            keep.append(x); last = x
    ax.set_xticks(keep)
    ax.set_xticklabels([f"{int(x):,}" for x in keep], fontsize=8)
    ax.set_xticks([], minor=True)
    ax.set_xlabel("blocks allocated, at a fixed 320k points", fontsize=8,
                  color=pal["muted"])
    ax.set_ylabel("allocate (ms)", fontsize=8, color=pal["muted"])
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5])
    ax.set_yticks([], minor=True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(min(xs) * 0.6, max(xs) * 3.2)
    style(ax, pal)
    fig.tight_layout()
    return fig


def fig_counters(counters_csv, pal):
    """Where the Rust gap actually lived, and what the one-line fix moved.

    Each metric is normalised to CUDA C++ = 1.0 so that quantities in different
    units share one axis; a dual axis would be the alternative and is never the
    right answer.
    """
    rows = {r["metric"]: r for r in csv.DictReader(open(counters_csv))}
    want = [
        ("duration", "kernel duration", False),
        ("long_scoreboard stall", "memory-wait stall", False),
        ("L2 sectors", "L2 sectors moved", False),
        ("L1 sector hit rate", "L1 hit rate", True),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ok = []
    for metric, label, higher_is_better in want:
        r = rows.get(metric)
        if not r or not r["rust_after"].strip():
            continue
        base = float(r["cuda_cpp"])
        ok.append((label, float(r["rust_before"]) / base,
                   float(r["rust_after"]) / base, higher_is_better))

    for i, (label, before, after, _) in enumerate(ok):
        ax.plot([before, after], [i, i], color=pal["rust"], lw=2.5, alpha=.35,
                solid_capstyle="round", zorder=2)
        ax.plot([before], [i], "o", color=pal["bg"], ms=9, zorder=3,
                markeredgecolor=pal["rust"], markeredgewidth=2)
        ax.plot([after], [i], "o", color=pal["rust"], ms=9, zorder=4,
                markeredgecolor=pal["bg"], markeredgewidth=1.5)

    ax.axvline(1.0, color=pal["rule"], lw=1, zorder=1)
    ax.annotate("CUDA C++", (1.0, len(ok) - 0.42), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=pal["muted"])
    ax.set_yticks(range(len(ok)))
    ax.set_yticklabels([o[0] for o in ok])
    ax.set_ylim(-0.6, len(ok) - 0.3)
    ax.set_xlim(0.35, 1.85)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xlabel("relative to CUDA C++ · hollow = before the fix, "
                  "filled = after", fontsize=8, color=pal["muted"])
    style(ax, pal)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig


def to_svg(fig, path, light):
    """Write an SVG whose colours are CSS variables rather than fixed hexes."""
    fig.savefig(path, format="svg", transparent=True, bbox_inches="tight")
    s = path.read_text()
    for name, hexval in light.items():
        target = "var(--bg)" if name == "bg" else VARS[name]
        s = s.replace(hexval, target).replace(hexval.upper(), target)
    # matplotlib writes an explicit white page under the axes; drop it.
    s = s.replace("#ffffff", "none").replace("#FFFFFF", "none")
    s = s.replace('<svg ', '<svg role="img" ', 1)
    path.write_text(s)


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    sweep = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
        root / "results/sweep_20260807_124145/sweep.csv"
    counters = root / "results/counters/allocate_counters.csv"
    pdf_dir = root / "paper/figures"
    svg_dir = root / "docs/figures"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    med, meta = load(sweep)
    builders = [
        ("stages", lambda p: fig_stages(med, meta, p)),
        ("floor", lambda p: fig_floor(med, meta, p)),
        ("axes", lambda p: fig_axes(med, meta, p)),
        ("counters", lambda p: fig_counters(counters, p)),
    ]
    for name, build in builders:
        # The paper is printed on white, so it takes the light palette only.
        f = build(LIGHT)
        f.savefig(pdf_dir / f"{name}.pdf", transparent=True, bbox_inches="tight")
        plt.close(f)
        # The page re-themes at read time, so its SVG carries variables.
        f = build(LIGHT)
        to_svg(f, svg_dir / f"{name}.svg", LIGHT)
        plt.close(f)
        print(f"  {name}: paper/figures/{name}.pdf  docs/figures/{name}.svg")


if __name__ == "__main__":
    main()
