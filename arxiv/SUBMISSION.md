# arXiv submission

Published as **[arXiv:2608.08287](https://arxiv.org/abs/2608.08287)**, submitted
8 August 2026 and announced the same weekend, well inside the two business days
arXiv quotes.

This file was written before the submission and predicted several things wrong.
It is now a record of what actually happened, because its remaining value is for
the next one: a v2 of this paper, or a different paper.

## What was submitted

| Field | Value |
|---|---|
| Identifier | arXiv:2608.08287v1 |
| Primary | **cs.CV** |
| Cross-lists | cs.PL, cs.DC, cs.PF |
| Licence | CC BY 4.0 |
| Compiler | pdflatex, TeX Live 2025 (arXiv default) |
| Package | `arxiv-what-irregularity-costs.tar.gz`, 19 files, 2.5 MB |

Comments field as filed:

> 23 pages, 5 figures, 4 tables. Includes a correctness result for TSDF fusion
> implementations: at hash load factors reached by ordinary depth trajectories,
> the Triton implementation silently discards blocks. Code, raw measurement CSVs
> and an interactive viewer:
> https://github.com/realitymatrix/what-irregularity-costs

ACM classes: `D.3.4; I.4.8; C.1.2`. Report number, journal reference, external
DOI and MSC class left blank.

## Rebuilding the package

```bash
tools/mk_arxiv.sh
```

It assembles the tarball, then unpacks it into an empty directory and builds
there. Building in the directory it was assembled in proves nothing, because
anything the package forgot is still sitting next to it. The check is 23 pages,
zero unresolved references, and text identical to the committed
`paper/main.pdf`. Figures are copied from what `main.tex` actually asks for, so
one added cannot be forgotten and one removed stops shipping.

## What this file got wrong, and what is true

**The package HAS been compiled with arXiv's toolchain.** An earlier version of
this note said it had not, because only `tectonic` was installed. It was later
verified against real TeX Live in Docker: three `pdflatex` passes, exit 0 each,
zero undefined references, 23 pages, US Letter, output matching the tectonic
build apart from line breaking. Tested on TeX Live 2026 where arXiv defaults to
2025; nothing here is version-sensitive.

```bash
docker run --rm -v "$PWD":/w -w /w texlive/texlive:latest \
  sh -c 'for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error main.tex; done'
```

**arXiv prefers the `.bib`, not the `.bbl`.** This note previously claimed the
opposite. The v1.5 file scan pre-selects `main.bbl` for deletion and says a
`.bib` is preferred, so it can regenerate the bibliography if a style changes.
Accepting that default is correct and was verified first: with `main.bbl`
removed, BibTeX regenerates all 9 entries, zero undefined citations, and the
output is byte-identical.

Ship **both** files regardless. Tectonic ignores a pre-existing `.bbl` and
re-runs BibTeX, so a package carrying only the `.bbl` builds locally with every
citation as `[?]` and no bibliography at all. Both files satisfies either
toolchain, and arXiv deletes the one it does not want.

**The abstract field caps at 1920 characters, counted as the browser submits
them.** A textarea converts each newline to CRLF on submission, so every
paragraph break costs one character more than the file contains. A 1,916
character file was rejected at 1,922. Count with:

```python
len(text.replace("\n", "\r\n"))
```

**Endorsement is per subject class, and cross-listing is not blocked by it.**
Being registered to a group is not the same as being endorsed for a class in it.
cs.PL was refused as a primary, but adding it as a cross-list after choosing a
primary the account was endorsed for went through without an endorser.

**`\graphicspath{{figures/}}` with a subdirectory is fine.** The scan resolved
every file and reported all of them as used by `main.tex`.

## On the primary category

The paper's contribution is a language and compiler result, so cs.PL is its
methodological home and is cross-listed. cs.CV is the primary because the
finding with the shortest path to harm is a correctness one about TSDF fusion,
and the people who need it are writing fusion pipelines.

If that classification changes, change the header comment in `paper/main.tex`
too. arXiv distributes the LaTeX source, so a listing whose own source argues
against its category is visible to anyone who downloads it. An earlier version
of that comment said "NOT cs.CV" and would have shipped that way.

## v1 is permanent

arXiv keeps every version publicly accessible, and withdrawal replaces a paper
with a withdrawal notice rather than removing it. The numbers in v1 come from
`results/sweep_20260807_202302/`. Widening the real-data axis has already
changed a conclusion once: with a single scene Triton's real-data cost looked
mid-range, and with nine cells it is at the worse end. A third environment or a
physical sensor would move it again, and that is a v2, not an erratum.
