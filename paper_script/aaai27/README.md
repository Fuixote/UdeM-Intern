# AAAI-27 Paper Draft

This directory is the paper-writing root for the AAAI-27 submission.

## Source Layout

- `main.tex`: AAAI-27 anonymous submission entrypoint.
- `sections/`: paper sections split into small editable files.
- `figures/`: paper-local figure copies.
- `tables/`: paper-local table inputs.
- `aaai2027.sty`, `aaai2027.bst`, `ReproducibilityChecklist.tex`: official files from `AuthorKit27.zip`.
- `references.bib`: paper bibliography.

## Build

Run from this directory:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

Clean generated LaTeX files:

```bash
latexmk -C main.tex
```

## Sync Toy Artifacts

After regenerating the decision-analysis toy examples, refresh the paper-local
copies from the repo root:

```bash
cp surrogate_experiment_results/decision_analysis/plots/toy_examples/toy_regret_comparison.png \
  paper_script/aaai27/figures/toy_regret_comparison.png
cp surrogate_experiment_results/decision_analysis/plots/toy_examples/toy_parametric_epsilon_curve.png \
  paper_script/aaai27/figures/toy_parametric_epsilon_curve.png
cp surrogate_experiment_results/decision_analysis/results/toy_examples/toy_summary_for_paper.tex \
  paper_script/aaai27/tables/toy_summary_for_paper.tex
```
