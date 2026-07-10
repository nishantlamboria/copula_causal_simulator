# Adaptive non-simplified conditional copulas

This patch adds a production implementation of a piecewise-constant
conditional copula for every selected two-parent D-vine order

\[
A-B-X.
\]

The first-tree copulas remain global.  The second-tree copula of
\(F(A\mid B)\) and \(F(X\mid B)\) can vary across equal-frequency bins of
\(U_B\).

## Selection procedure

1. Select the better parent order as before.
2. Make a deterministic train-validation split for conditional-model
   selection.
3. Select one global conditional family and rotation on the training data.
4. Fit the same family and rotation in candidate quantile-bin counts
   \(K\in\{1,2,3,4,5,6\}\).
5. Compare mean held-out conditional log-likelihood.
6. Replace the simplifying assumption only when the best multi-bin model
   improves on \(K=1\) by at least the configured threshold.
7. Refit the selected representation on all observations.

The family and rotation remain fixed across bins.  Only parameters vary.
This is deliberately more stable than selecting an unrelated family in every
bin.

## Apply the patch

Extract the patch ZIP into the repository root while preserving directories.
Then run:

```powershell
python -m pytest -q
```

## Configuration

Each dataset config contains:

```yaml
copula:
  conditional_model_strategy: adaptive_quantile_bins
  conditional_candidate_bin_counts: [1, 2, 3, 4, 5, 6]
  conditional_minimum_bin_size: 75
  conditional_validation_fraction: 0.20
  conditional_selection_seed: 42
  conditional_minimum_score_improvement: 0.005
  conditional_fixed_family_across_bins: true
```

To reproduce the previous model, set:

```yaml
conditional_model_strategy: simplified
```

## Rebuild one dataset

```powershell
python scripts\build_conditional_vine_library.py `
  --config configs\datasets\sachs.yaml
```

Command-line overrides are available:

```powershell
python scripts\build_conditional_vine_library.py `
  --config configs\datasets\sachs.yaml `
  --conditional-model-strategy adaptive_quantile_bins `
  --conditional-candidate-bin-counts 1 2 3 4 5 6 `
  --conditional-minimum-bin-size 75 `
  --conditional-validation-fraction 0.20 `
  --conditional-selection-seed 42 `
  --conditional-minimum-score-improvement 0.005
```

## Rebuild all three datasets and create reports

```powershell
powershell -ExecutionPolicy Bypass -File `
  scripts\rebuild_adaptive_conditional_libraries.ps1
```

For a quick smoke run:

```powershell
powershell -ExecutionPolicy Bypass -File `
  scripts\rebuild_adaptive_conditional_libraries.ps1 `
  -SkipTests `
  -Limit 5
```

## Output metadata

Each successful record now includes:

- `conditional_model_type`: `simplified` or `quantile_binned`;
- `selected_number_bins`;
- global conditional family and rotation;
- candidate validation scores and failure diagnostics;
- simplified and selected validation scores;
- improvement over the simplified representation;
- full-data bin edges and bin counts;
- one fitted model file per selected bin;
- empirical and fitted conditional Kendall's tau by bin.

Old metadata remains loadable.  When the new fields are absent, the generator
uses the original single conditional copula.

## Diagnostic report

After rebuilding a dataset:

```powershell
python scripts\report_conditional_model_selection.py `
  --summary reports\tables\sachs\conditional_vine_summary_indegree2.csv `
  --output-dir reports\conditional_models\sachs
```

The report includes:

- distribution of selected bin counts;
- held-out score improvement over the simplifying assumption;
- a compact mechanism-level table;
- flattened conditional-tau diagnostics;
- per-mechanism empirical and fitted conditional-tau curves.

## Interpretation

Selecting multiple bins means only that a piecewise conditional-copula model
provided a sufficiently better held-out statistical fit.  It does not change
the DAG, identify a causal order, or itself establish interventional validity.

Equal-frequency binning is a transparent first relaxation of the simplifying
assumption.  It can still be discontinuous at bin boundaries and becomes
unsuitable when the conditioning dimension grows.  Smooth varying-parameter
copulas remain a later extension.
