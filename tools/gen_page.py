#!/usr/bin/env python3
"""Generate the project page from the same CSVs the paper's tables come from.

The page and the paper must never disagree, so neither is written by hand.

Usage:  tools/gen_page.py results/sweep_YYYYmmdd_HHMMSS/sweep.csv docs/index.html
"""
import csv, collections, statistics, sys, pathlib

GPU = {"0": "RTX 5070 Ti", "1": "RTX 5060"}
TOTALS = [("tartan-warm", "TartanAir, warm volume"), ("tartan-cold", "TartanAir, cold volume"),
          ("plane-320k", "Plane, 320k points"), ("pts-1280k", "Sphere, 1.28M points"),
          ("base", "Sphere, 320k points")]
CELLS = [("pts-20k", "points"), ("pts-80k", "points"), ("base", "points"),
         ("pts-720k", "points"), ("pts-1280k", "points"), ("r-0.25", "extent"),
         ("r-1.0", "extent"), ("r-2.0", "extent"), ("plane-320k", "shape"),
         ("tartan-cold", "real"), ("tartan-warm", "real")]


def load(path):
    acc, meta = collections.defaultdict(list), {}
    for r in csv.DictReader(open(path)):
        acc[(r["device"], r["cell"], r["arm"], r["stage"])].append(float(r["p50_ms"]))
        meta[r["cell"]] = (int(r["points"]), float(r["load_factor"]))
    return {k: statistics.median(v) for k, v in acc.items()}, meta


def ratio_range(m, stage, arm):
    v = []
    for (dev, cell, a, st), val in m.items():
        if st == stage and a == arm:
            b = m.get((dev, cell, "A3-cuda", st))
            if b:
                v.append(val / b)
    return min(v), max(v)


def build(m, meta):
    tot = []
    for cell, label in TOTALS:
        for dev in ("0", "1"):
            t = {a: m.get((dev, cell, a, "allocate"), 0) + m.get((dev, cell, a, "update"), 0)
                 for a in ("A3-cuda", "A4-rust", "A5-triton")}
            if not t["A3-cuda"]:
                continue
            tot.append(f"""<tr><td>{label}</td><td class=g>{GPU[dev]}</td>
<td class=n>{t['A3-cuda']:.3f}</td><td class=n>{t['A4-rust']:.3f}</td>
<td class="n r">{t['A4-rust']/t['A3-cuda']:.2f}&times;</td>
<td class=n>{t['A5-triton']:.3f}</td>
<td class="n r">{t['A5-triton']/t['A3-cuda']:.2f}&times;</td></tr>""")

    cel, prev = [], None
    for cell, axis in CELLS:
        b = m.get(("0", cell, "A3-cuda", "allocate"))
        if not b:
            continue
        cls = ' class="sep"' if prev and axis != prev else ""
        prev = axis
        pts, lf = meta[cell]
        cel.append(f"""<tr{cls}><td><code>{cell}</code></td><td class=g>{axis}</td>
<td class=n>{pts:,}</td><td class=n>{lf:.3f}</td>
<td class="n r">{m[('0',cell,'A4-rust','allocate')]/b:.2f}</td>
<td class="n r">{m[('0',cell,'A5-triton','allocate')]/b:.1f}</td></tr>""")

    st = {(s, a): ratio_range(m, s, a) for s in ("allocate", "update")
          for a in ("A4-rust", "A5-triton")}
    return "\n".join(tot), "\n".join(cel), st


def main():
    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/sweep_20260807_124145/sweep.csv")
    out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "docs/index.html")
    m, meta = load(src)
    totals, cells, st = build(m, meta)
    tpl = (pathlib.Path(__file__).parent / "page_template.html").read_text()
    html = (tpl.replace("{{TOTALS}}", totals).replace("{{CELLS}}", cells)
               .replace("{{ALLOC_RUST}}", f"{st[('allocate','A4-rust')][0]:.2f}&ndash;{st[('allocate','A4-rust')][1]:.2f}&times;")
               .replace("{{ALLOC_TRITON}}", f"{st[('allocate','A5-triton')][0]:.1f}&ndash;{st[('allocate','A5-triton')][1]:.1f}&times;")
               .replace("{{UPD_RUST}}", f"{st[('update','A4-rust')][0]:.2f}&ndash;{st[('update','A4-rust')][1]:.2f}&times;")
               .replace("{{UPD_TRITON}}", f"{st[('update','A5-triton')][0]:.1f}&ndash;{st[('update','A5-triton')][1]:.1f}&times;")
               .replace("{{SOURCE}}", src.parent.name))
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} from {src}")


if __name__ == "__main__":
    sys.exit(main())
