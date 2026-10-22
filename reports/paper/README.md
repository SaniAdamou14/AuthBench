# Technical report

`main.tex` is the AuthBench report, version 2: written against the wider-window
LANL run ([`../lanl/wider_window_results/RUN.md`](../lanl/wider_window_results/RUN.md)),
with the single-day run of [`../lanl/RUN.md`](../lanl/RUN.md) in an appendix.
Version 1 (single-day run) is archived on Zenodo,
[10.5281/zenodo.23159768](https://doi.org/10.5281/zenodo.23159768); its two
errors are listed in the correction notice at the top of the repository README.

## Where every number comes from

Nothing in the report is typed by hand. `scripts/paper_assets.py` reads the
committed run outputs and writes `generated/`:

| File | Content |
|---|---|
| `numbers.tex` | every number quoted in the text, as LaTeX macros |
| `table_*.tex` | the rows of each results table |
| `figure_first_detection.pdf` | budget for first detection against ROC-AUC and AUC-PR |

CI reruns the script and fails if a generated `.tex` file differs from the
committed copy, so the text cannot drift from the runs it describes.

```bash
python scripts/paper_assets.py
```

## Building

The bibliography uses `biblatex` with the APA 7th edition style
(`biblatex-apa`) and the `biber` backend. Every entry of `references.bib` was
checked against a primary record (Crossref, DBLP, arXiv or the publisher).

```bash
cd reports/paper
latexmk -pdf main.tex      # runs pdflatex and biber as many times as needed
```

Packages beyond a standard TeX Live or MiKTeX install: `biblatex-apa`,
`siunitx`, `threeparttable`, `cleveref`; MiKTeX installs them on first use.

## Submitting to arXiv

1. **Endorsement first.** Since 21 January 2026, a first submission to `cs.CR`
   needs a personal endorsement from an established author in that category;
   an institutional address no longer suffices. Settle it before the text is
   final.
2. Primary category `cs.CR`; cross-list `cs.LG`.
3. Upload the source: `main.tex`, `references.bib`, `generated/`, and the
   `.bbl` produced by biber (arXiv does not run biber).
4. CC BY 4.0, consistent with the Apache-2.0 code licence.
5. The abstract in the form must match the document.

## Deliberately not wired into DVC

There is no `report` stage in `dvc.yaml`: it would make `dvc repro` require a
TeX toolchain for every clone. The report is a document about the runs, not a
stage of them.
