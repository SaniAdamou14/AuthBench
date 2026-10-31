# Technical report

`main.tex` is the AuthBench report, version 2: written against the full LANL
run of 22 October 2026
([`../runs/20261022T102638Z/RUN.md`](../runs/20261022T102638Z/RUN.md)), both
protocols, with the wider-window run of September
([`../lanl/wider_window_results/RUN.md`](../lanl/wider_window_results/RUN.md))
compared in an appendix on the same temporal partitions.
Version 1 (single-day run) is archived on Zenodo,
[10.5281/zenodo.23159768](https://doi.org/10.5281/zenodo.23159768); its two
errors are listed in the correction notice at the top of the repository README.
Version 2 is archived as
[10.5281/zenodo.23687498](https://doi.org/10.5281/zenodo.23687498). Since then,
`main.tex` carries one correction that the archived PDF does not: the favourable
end of the tie bracket is now the exact maximum over tie orders, so pair rarity
(M2a) at 10 alerts per day reads 34.0% instead of 32.1% (harmonised) and 38.5%
instead of 32.7% (temporal). Nothing else in the report changes; cite the
archived version 2 until a new version is archived.

## Where every number comes from

No result of a run is typed by hand: the text quotes them through macros, and the
tables and figures are generated. What is typed are constants of the protocol (the
budgets, the 749 red-team lines, the staffing inputs) and figures published by other
works. `scripts/paper_assets.py` reads the committed run outputs and writes `generated/`:

| File | Content |
|---|---|
| `numbers.tex` | every number quoted in the text, as LaTeX macros |
| `table_*.tex` | the rows of each results table |
| `figure_first_detection.pdf` | budget for first detection against ROC-AUC and AUC-PR, both protocols |
| `figure_campaign_recall.pdf` | campaign recall against the daily budget, from the positives ledger |

The script also checks the run against itself before writing (first detections in the ledger
against the metrics, the simulated random-ranking null against its exact formula, the Euler
recounts against the published counts) and stops on any disagreement.

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
