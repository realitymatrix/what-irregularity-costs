# arXiv submission package

`arxiv-what-irregularity-costs.tar.gz`, 2.5 MB, 18 files.

Built and verified by unpacking the tarball into an empty directory and
compiling there, so nothing leaks in from the source tree. Output is 23 pages,
US Letter, zero unresolved references, and its extracted text is byte-identical
to the committed `paper/main.pdf`.

## Regenerating

```bash
tools/gen_figures.py results/sweep_*/sweep.csv   # figures
paper/gen_tables.py  results/sweep_*/sweep.csv   # tables
tools/mk_arxiv.sh                                # package
```

## Form fields

| Field | Value |
|---|---|
| Title | What Irregularity Costs: CUDA C++, Rust, and Triton on a Hash-Blocked GPU Workload |
| Author | Petr Korolev (Spacial Intelligence Labs) |
| Primary | cs.PL |
| Cross-list | cs.DC, cs.PF |
| Licence | pick one; CC BY 4.0 matches the repo's MIT posture |

Abstract is in `abstract.txt`, already stripped of LaTeX markup for pasting
into the web form.

Suggested Comments field:

> 23 pages, 7 figures. Code, raw measurement CSVs and an interactive viewer:
> https://github.com/realitymatrix/what-irregularity-costs

## Notes for the submitter

**Both `main.bbl` and `refs.bib` are included, deliberately.** arXiv does not
run BibTeX and uses the `.bbl`. Tectonic does the opposite: it ignores a
pre-existing `.bbl` and re-runs BibTeX, so a package with only the `.bbl` built
locally with every citation rendered as `[?]` and no bibliography at all. That
is a difference between the two toolchains rather than a defect in either, and
shipping both files satisfies each of them.

**This package has not been compiled with arXiv's actual toolchain.** Only
`tectonic` is installed here, and arXiv runs TeX Live with `pdflatex`. The
package uses nothing exotic (`geometry`, `booktabs`, `amsmath`, `graphicx`,
`listings`, `xcolor`, `hyperref`, `microtype`), figures are PDF 1.4 and 8-bit
PNG, and there are no absolute paths, so it should compile. arXiv shows you its
own build log before you commit to announcing; read it rather than trusting
this paragraph.

**`\graphicspath{{figures/}}` with a subdirectory is supported by arXiv.** If
their build objects, flatten `figures/` into the root and drop that line.

**First submission to cs.PL may need an endorsement.** That is the step with a
human in the loop, so check it before setting aside time for the rest.

**v1 is permanent.** arXiv keeps every version publicly accessible and
withdrawal replaces a paper with a withdrawal notice rather than removing it.
The numbers here come from `results/sweep_20260807_202302/`, and widening the
real-data axis has already changed a conclusion once: with a single scene,
Triton's real-data cost looked mid-range, and with nine cells it is at the worse
end. A third environment or a physical sensor would move it again.
