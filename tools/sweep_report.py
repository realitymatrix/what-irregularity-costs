#!/usr/bin/env python3
"""Turn a sweep CSV into the ratio tables the paper actually needs.

The raw CSV is per (device, cell, arm, stage). What the argument turns on is
the RATIO between arms within a cell, and how that ratio moves along each axis.
A ratio that is constant across every axis is a property of the languages; one
that swings is a property of the workload, and saying which is which is the
whole point of running a sweep rather than one scene.

Stdlib only, deliberately: this has to run six months from now without a
resolvable environment.

Usage:
    tools/sweep_report.py results/sweep_*/sweep.csv
    tools/sweep_report.py results/sweep_*/sweep.csv --stage allocate
"""
import argparse
import csv
import sys
from collections import defaultdict

BASE = "A3-cuda"  # the reference arm every ratio is taken against


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def collapse(rows, metric):
    """Reduce repeated passes to one value per (device, cell, arm, stage).

    Reports the MEDIAN across passes, and the spread between the best and worst
    pass, because a single pass understates how unstable a cell is. Cell r-2.0
    reported a p50 of 0.047 ms in one pass and 0.060 in each of three reruns,
    a 30% swing that no within-pass statistic separated from ordinary width.
    Anything above 15% is flagged: it is larger than several of the language
    effects being argued about, and a reader is entitled to know which rows
    cannot carry a 10% claim.
    """
    acc = defaultdict(list)
    proto = {}
    for r in rows:
        k = (r["device"], r["cell"], r["arm"], r["stage"])
        acc[k].append(r[metric])
        proto.setdefault(k, r)
    out = {}
    for k, vs in acc.items():
        lo, hi = min(vs), max(vs)
        out[k] = {"v": med(vs), "n_runs": len(vs),
                  "spread": (hi / lo) if lo > 0 else float("inf"), "row": proto[k]}
    return out


def load(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    for r in rows:
        for k in ("run", "points", "pool_blocks", "blocks", "batch", "n", "sm_count",
                  "scratch_slots"):
            if r.get(k):
                r[k] = int(r[k])
        for k in ("p50_ms", "p95_ms", "p99_ms", "min_ms", "mean_ms", "load_factor", "voxel_m"):
            if r.get(k):
                r[k] = float(r[k])
    return rows


def table(rows, stage, metric, coll):
    """One block per device: cells down, arms across, with ratios to A3.

    Values are medians across passes; see collapse().
    """
    by_dev = defaultdict(list)
    for r in rows:
        if r["stage"] == stage:
            by_dev[(r["device"], r["device_name"], r["sm_count"])].append(r)

    for (dev, name, sms), drows in sorted(by_dev.items()):
        cells = []
        for r in drows:
            if r["cell"] not in [c[0] for c in cells]:
                cells.append((r["cell"], r["axis"], r["points"], r["load_factor"]))
        arms = sorted({r["arm"] for r in drows})
        arms = [BASE] + [a for a in arms if a != BASE]

        print(f"\n### {stage} / {metric} / device {dev}: {name} ({sms} SMs)\n")
        head = f"{'cell':<12} {'axis':<11} {'points':>8} {'load':>6} "
        head += " ".join(f"{a:>11}" for a in arms)
        head += "   " + " ".join(f"{'x/' + BASE[:2]:>8}" for a in arms if a != BASE)
        print(head)
        print("-" * len(head))

        for cell, axis, pts, lf in cells:
            vals, spreads = {}, {}
            for a in arms:
                c = coll.get((dev, cell, a, stage))
                if c:
                    vals[a] = c["v"]
                    spreads[a] = c["spread"]
            line = f"{cell:<12} {axis:<11} {pts:>8} {lf:>6.3f} "
            line += " ".join(f"{vals.get(a, float('nan')):>11.4f}" for a in arms)
            # Flag a cell whose own distribution is wide. A p95 far above p50
            # means the cell is bimodal and its p50 is not a stable summary:
            # r-2.0 measured A3 at 0.047 in one sweep and 0.060 in three
            # consecutive reruns, a 30% swing larger than several of the effects
            # under discussion.
            wide = [a for a in arms if spreads.get(a, 1.0) > 1.15]
            line += "   "
            base = vals.get(BASE)
            for a in arms:
                if a == BASE:
                    continue
                if base and a in vals and base > 0:
                    line += f"{vals[a] / base:>8.2f}"
                else:
                    line += f"{'-':>8}"
                line += " "
            if wide:
                line += "  RUN-TO-RUN>15%(" + ",".join(a.split("-")[0] for a in wide) + ")"
            print(line)


def axis_summary(rows, stage, metric, coll):
    """How stable is each arm's ratio to A3 along each axis?

    This is the question the sweep was built to answer. A small spread means
    the ratio is a language property; a large one means the headline number
    was a property of whichever cell happened to be measured first.
    """
    print(f"\n### ratio stability, {stage} / {metric}\n")
    print(f"{'device':>6} {'arm':<11} {'axis':<11} {'cells':>5} {'min':>8} {'max':>8} "
          f"{'spread':>8}")
    print("-" * 62)

    key = defaultdict(dict)
    for (dev, cell, arm, st), c in coll.items():
        if st == stage:
            key[(dev, cell)][arm] = c["v"]

    per = defaultdict(list)
    axis_of = {}
    for r in rows:
        axis_of[(r["device"], r["cell"])] = r["axis"]
    for (dev, cell), arms in key.items():
        base = arms.get(BASE)
        if not base:
            continue
        for a, v in arms.items():
            if a == BASE:
                continue
            per[(dev, a, axis_of[(dev, cell)])].append(v / base)

    for (dev, arm, axis), vs in sorted(per.items()):
        lo, hi = min(vs), max(vs)
        spread = hi / lo if lo > 0 else float("inf")
        flag = "  <-- workload-dependent" if spread > 1.5 else ""
        print(f"{dev:>6} {arm:<11} {axis:<11} {len(vs):>5} {lo:>8.2f} {hi:>8.2f} "
              f"{spread:>8.2f}x{flag}")

    print("\n  spread = max ratio / min ratio within that axis. Near 1.0 means the")
    print("  arm's disadvantage is a property of the language; a large spread means")
    print("  it is a property of the workload and must be reported per-regime.")


def device_scaling(rows, stage, metric, coll):
    """Same cell, different device. Tests whether the ranking is machine-width
    dependent. Both local cards are sm_120, so codegen is fixed and only SM
    count and bandwidth vary; a ranking that flips here is significant."""
    devs = sorted({(r["device"], r["sm_count"], r["device_name"]) for r in rows})
    if len(devs) < 2:
        return
    print(f"\n### device scaling, {stage} / {metric}\n")
    lo = min(devs, key=lambda d: d[1])
    hi = max(devs, key=lambda d: d[1])
    print(f"  narrow: device {lo[0]} {lo[2]} ({lo[1]} SMs)")
    print(f"  wide:   device {hi[0]} {hi[2]} ({hi[1]} SMs)")
    print(f"  ratio of SM counts: {hi[1] / lo[1]:.2f}x\n")

    print(f"{'cell':<12} {'arm':<11} {'narrow':>9} {'wide':>9} {'speedup':>9}")
    print("-" * 53)
    vals = defaultdict(dict)
    batches = defaultdict(dict)
    for (dev, cell, arm, st), c in coll.items():
        if st == stage:
            vals[(cell, arm)][dev] = c["v"]
            batches[cell][dev] = c["row"]["batch"]
    for (cell, arm), d in sorted(vals.items()):
        if lo[0] in d and hi[0] in d and d[hi[0]] > 0:
            # A cell whose BATCH differs between devices is not a valid
            # comparison: batching is what amortises the completion wait, so a
            # cell measured at batch 4 on one card and 8 on the other is
            # measuring two different things. The harness reduces batch
            # automatically when a cell will not fit, which is right, but it
            # makes the cross-device row meaningless and that must be said.
            b = batches[cell]
            note = ""
            if b.get(lo[0]) != b.get(hi[0]):
                note = f"   NOT COMPARABLE (batch {b.get(lo[0])} vs {b.get(hi[0])})"
            print(f"{cell:<12} {arm:<11} {d[lo[0]]:>9.4f} {d[hi[0]]:>9.4f} "
                  f"{d[lo[0]] / d[hi[0]]:>8.2f}x{note}")

    print("\n  speedup = narrow / wide. An arm bound by issue rate should track the")
    print("  SM ratio; one bound by contention on a few hot addresses should not.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--stage", default="allocate,update")
    ap.add_argument("--metric", default="p50_ms")
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        print("no rows", file=sys.stderr)
        return 1

    devs = {(r["device"], r["device_name"]) for r in rows}
    cells = {r["cell"] for r in rows}
    print(f"# sweep report: {args.csv}")
    print(f"  {len(rows)} rows | {len(devs)} device(s) | {len(cells)} cell(s)")
    print(f"  reference arm: {BASE} | metric: {args.metric}")

    coll = collapse(rows, args.metric)
    n_runs = len({r["run"] for r in rows}) if "run" in rows[0] else 1
    print(f"  passes: {n_runs}" + ("  (values are medians across passes)" if n_runs > 1
                                   else "  <-- SINGLE PASS, run-to-run spread unknown"))
    for stage in args.stage.split(","):
        table(rows, stage, args.metric, coll)
        axis_summary(rows, stage, args.metric, coll)
        device_scaling(rows, stage, args.metric, coll)
    return 0


if __name__ == "__main__":
    sys.exit(main())
