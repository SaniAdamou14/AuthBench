# Technical report

`main.tex` — the AuthBench technical report, written against the committed run in
[`../lanl/RUN.md`](../lanl/RUN.md).

Every figure in it is traceable to committed output or to the docs:
`../lanl/RUN.md` (split, results table, pairwise comparisons),
`../../docs/dataset-notes.md` (749 raw rows, 715 after deduplication),
`../../docs/methodology.md` (protocol, bootstrap floors, transductivity of ECOD),
`../../docs/limitations.md` (one-day history, cold-start sentinel),
`../../README.md` (rule discrimination table, scaling figures, timings).

**Nothing in it exceeds what those files establish.** Where the run is ambiguous
— M0b's undetermined budget row, the twelve significant comparisons that are
bounds rather than measurements — the report says so rather than rounding in its
own favour.

## Building

No TeX distribution is required to read it. To produce a PDF:

```bash
pdflatex main.tex && pdflatex main.tex   # twice, for cross-references
```

The document uses only standard packages (`amsmath`, `amssymb`, `booktabs`,
`graphicx`, `hyperref`, `url`, `geometry`) and a plain `article` class, so it
compiles on any TeX Live or MiKTeX install, and on Overleaf with no
configuration.

**One figure is included by relative path** —
`../lanl/figures/campaign_recall_vs_budget.png` — so that the paper always shows
the committed output rather than a copy that can drift from it. This works for a
local build. It does **not** work for an arXiv upload, which must be
self-contained: copy the PNG next to `main.tex` and change the path to
`campaign_recall_vs_budget.png` before packaging. Do it at submission time, not
before, so the working copy keeps pointing at the run.

If no local TeX is installed, Overleaf is the shortest path: create a blank
project, upload `main.tex`, compile. It also produces the archive arXiv expects.

## Submitting to arXiv

1. **Endorsement comes first, not last.** Since 21 January 2026, arXiv no longer
   accepts an institutional email address as sufficient credential for a first
   submission to a category. A first submission to `cs.CR` therefore requires an
   endorsement from an established author in that category. This is an
   administrative dependency with a human in the loop and unknown latency —
   settle it before the text is final, not after.
2. Primary category `cs.CR` (Cryptography and Security); cross-list `cs.LG`
   (Machine Learning) is defensible given the evaluation-methodology content.
3. Upload the LaTeX source, not a PDF — arXiv prefers source and will compile it.
4. Choose a license (CC BY 4.0 is consistent with the Apache-2.0 code licence).
5. The abstract in the submission form must match the one in the document.

## Deliberately not wired into DVC

There is no `report` stage in `dvc.yaml`, and adding one that depends on this
file would make `dvc repro` require a TeX toolchain for everyone who clones the
repository. The report is a document about the run, not a stage of it.
