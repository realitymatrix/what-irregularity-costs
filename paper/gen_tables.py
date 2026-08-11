#!/usr/bin/env python3
"""Generate the paper's results tables from the raw sweep CSVs.

Nothing in the paper is typed by hand. This project corrected published figures
five times (Triton 73x to 17.9x, extraction 1.825 ms to a PCIe measurement, the
spin's share of the gap, a fake device axis, an unreproducible single-pass p50),
and every one of those would have silently survived in a hand-copied table.

Usage:  paper/gen_tables.py results/sweep_YYYYmmdd_HHMMSS/sweep.csv
"""
import csv, collections, statistics, sys, pathlib

DEV = {"0": "RTX 5070 Ti", "1": "RTX 5060"}
ARM = {"A3-cuda": "CUDA C++", "A4-rust": "Rust", "A5-triton": "Triton"}


def load(path):
    acc = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["axis"] == "coldwarm":
            continue
        acc[(r["device"], r["cell"], r["arm"], r["stage"])].append(float(r["p50_ms"]))
    return {k: statistics.median(v) for k, v in acc.items()}


def total(m, dev, cell, arm):
    a, u = m.get((dev, cell, arm, "allocate")), m.get((dev, cell, arm, "update"))
    return None if a is None or u is None else a + u


def tex_escape(s):
    return s.replace("_", r"\_")


def table_totals(m, out):
    rows = [("tartan-warm", "RetroOffice P000, warm"),
            ("tartan-cold", "RetroOffice P000, cold"),
            ("diner-p000", "AmericanDiner P000, warm"),
            ("plane-320k", "Plane, 320k points"),
            ("pts-1280k", "Sphere, 1.28M points"),
            ("base", "Sphere, 320k points")]
    rows = [r for r in rows if any(k[1] == r[0] for k in m)]
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Workload & GPU & CUDA C++ & \multicolumn{2}{c}{Rust} & \multicolumn{2}{c}{Triton} \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r" & & ms & ms & $\times$ & ms & $\times$ \\", r"\midrule"]
    for cell, label in rows:
        for dev in ("0", "1"):
            t3, t4, t5 = (total(m, dev, cell, a) for a in ("A3-cuda", "A4-rust", "A5-triton"))
            if t3 is None:
                continue
            L.append(f"{tex_escape(label)} & {DEV[dev]} & {t3:.3f} & {t4:.3f} & "
                     f"{t4/t3:.2f} & {t5:.3f} & {t5/t3:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "totals.tex").write_text("\n".join(L) + "\n")
    return len(rows)


def table_by_stage(m, out):
    """Ratio range per stage per arm, across every recorded cell."""
    rng = collections.defaultdict(list)
    for (dev, cell, arm, stage), v in m.items():
        if arm == "A3-cuda":
            continue
        base = m.get((dev, cell, "A3-cuda", stage))
        if base:
            rng[(stage, arm)].append(v / base)
    L = [r"\begin{tabular}{llrr}", r"\toprule",
         r"Stage & Character & Rust / CUDA C++ & Triton / CUDA C++ \\", r"\midrule"]
    for stage, char in (("allocate", "irregular: probe, CAS, publication"),
                        ("update", "regular: walk and accumulate")):
        r4 = rng[(stage, "A4-rust")]
        r5 = rng[(stage, "A5-triton")]
        L.append(f"{stage} & {char} & {min(r4):.2f}--{max(r4):.2f} & "
                 f"{min(r5):.1f}--{max(r5):.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "by_stage.tex").write_text("\n".join(L) + "\n")


# Axis order is editorial; cell order inside an axis follows the data, so a
# cell added to the sweep appears here without anyone remembering to list it.
AXIS_ORDER = ["baseline", "points", "extent", "shape", "loadfactor", "real"]


def cell_order(meta):
    """Every cell present in the CSV, grouped by axis, ordered within it."""
    seen = {}
    for cell, (points, load, axis) in meta.items():
        seen.setdefault(axis, []).append((cell, points, load))
    out = []
    for axis in AXIS_ORDER + [a for a in sorted(seen) if a not in AXIS_ORDER]:
        for cell, _p, _l in sorted(seen.get(axis, []), key=lambda t: (t[1], t[0])):
            out.append((cell, axis))
    return out


def table_cells(m, meta, out):
    """Per-cell allocate ratios on the wide GPU. The trends are the evidence."""
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Cell & Axis & Points & Load & Rust & Triton \\",
         r"\midrule"]
    prev = None
    for cell, axis in cell_order(meta):
        if cell not in meta:
            continue
        if prev is not None and axis != prev:
            L.append(r"\addlinespace")
        prev = axis
        pts, lf, _axis = meta[cell]
        b = m.get(("0", cell, "A3-cuda", "allocate"))
        r4 = m.get(("0", cell, "A4-rust", "allocate"))
        r5 = m.get(("0", cell, "A5-triton", "allocate"))
        if not b:
            continue
        L.append(f"{tex_escape(cell)} & {axis} & {pts:,} & {lf:.3f} & "
                 f"{r4/b:.2f} & {r5/b:.1f} \\\\".replace(",", ","))
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "cells.tex").write_text("\n".join(L) + "\n")


def table_counters(out, src):
    """Profiler and disassembly counters for the allocate kernel.

    Kept as data rather than typed into the paper: these are the numbers that
    make the argument, and every one of them contradicts the intuition that a
    slower kernel is doing more work.
    """
    import csv as _csv
    rows = list(_csv.DictReader(open(src)))
    L = [r"\begin{tabular}{lrrr}", r"\toprule",
         r"Counter & CUDA C++ & Rust & Rust, fixed \\", r"\midrule"]
    for r in rows:
        if r["metric"] == "warp cycles per issued instruction":
            L.append(r"\addlinespace")
        after = r["rust_after"] or "--"
        def fmt(v):
            if v in ("", "--"):
                return "--"
            f = float(v)
            return f"{f:,.2f}" if "." in v else f"{int(f):,}"
        L.append(f"{tex_escape(r['metric'])} & {fmt(r['cuda_cpp'])} & "
                 f"{fmt(r['rust_before'])} & {fmt(after)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "counters.tex").write_text("\n".join(L) + "\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = pathlib.Path(sys.argv[1])
    out = pathlib.Path(__file__).parent / "tables"
    out.mkdir(exist_ok=True)
    meta = {}
    for r in csv.DictReader(open(src)):
        if r["axis"] == "coldwarm":
            continue
        meta[r["cell"]] = (int(r["points"]), float(r["load_factor"]), r["axis"])
    m = load(src)
    n = table_totals(m, out)
    table_by_stage(m, out)
    table_cells(m, meta, out)
    counters = src.parent.parent / "counters" / "allocate_counters.csv"
    if counters.exists():
        table_counters(out, counters)
    (out / "PROVENANCE.txt").write_text(
        f"Generated from {src}\nby paper/gen_tables.py. Do not edit by hand.\n")
    print(f"wrote {n} workload rows and the per-stage summary to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
