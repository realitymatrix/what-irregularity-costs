#!/usr/bin/env bash
# Build the arXiv submission tarball, and verify it away from the source tree.
#
# The verification is the point. A submission package that happens to compile
# in the directory it was assembled in proves nothing, because anything it
# forgot to include is still sitting next to it. So this unpacks the tarball
# into an empty directory, builds there, and compares the extracted text to the
# committed paper. If they differ, something is missing or stale.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER="$ROOT/paper"
OUT="$ROOT/arxiv"
PKG="$(mktemp -d)"
VERIFY="$(mktemp -d)"
trap 'rm -rf "$PKG" "$VERIFY"' EXIT

command -v tectonic >/dev/null || { echo "tectonic not found"; exit 1; }

# The .bbl has to be current, and only a build produces it.
( cd "$PAPER" && tectonic --keep-intermediates main.tex >/dev/null 2>&1 )
[ -s "$PAPER/main.bbl" ] || { echo "no main.bbl produced"; exit 1; }

mkdir -p "$PKG/figures" "$PKG/tables"
cp "$PAPER/main.tex" "$PAPER/main.bbl" "$PKG/"
# Both bibliography forms ship, deliberately. arXiv does not run BibTeX and
# reads the .bbl; tectonic ignores a pre-existing .bbl and re-runs BibTeX, so a
# package carrying only the .bbl builds here with every citation as [?]. That
# is a toolchain difference, not a defect, and shipping both satisfies each.
cp "$PAPER/refs.bib" "$PKG/"
cp "$PAPER"/tables/*.tex "$PKG/tables/"

# Copy exactly the figures the source asks for, so a figure deleted from the
# paper stops being shipped and one added cannot be forgotten.
grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$PAPER/main.tex" \
  | sed 's/.*{//;s/}//' | sort -u | while read -r f; do
    [ -f "$PAPER/figures/$f" ] || { echo "missing figure: $f"; exit 1; }
    cp "$PAPER/figures/$f" "$PKG/figures/$f"
done

# arXiv wants source. A stray main.pdf next to it invites the wrong questions.
rm -f "$PKG"/main.pdf "$PKG"/main.aux "$PKG"/main.log "$PKG"/main.out

mkdir -p "$OUT"
TARBALL="$OUT/arxiv-what-irregularity-costs.tar.gz"
tar czf "$TARBALL" -C "$PKG" .

( cd "$VERIFY" && tar xzf "$TARBALL" && tectonic main.tex >/dev/null 2>&1 )
[ -f "$VERIFY/main.pdf" ] || { echo "FAIL: the package does not build in isolation"; exit 1; }

bad=$(pdftotext "$VERIFY/main.pdf" - | grep -c '??' || true)
[ "$bad" = "0" ] || { echo "FAIL: $bad unresolved references in the isolated build"; exit 1; }

if ! diff -q <(pdftotext "$VERIFY/main.pdf" -) <(pdftotext "$PAPER/main.pdf" -) >/dev/null; then
    echo "FAIL: the isolated build differs from paper/main.pdf"
    exit 1
fi

pages=$(pdfinfo "$VERIFY/main.pdf" | awk '/^Pages/{print $2}')
printf 'ok  %s\n    %s pages, %s, %s files, text identical to paper/main.pdf\n' \
  "$TARBALL" "$pages" "$(du -h "$TARBALL" | cut -f1)" "$(tar tzf "$TARBALL" | grep -c '[^/]$')"
