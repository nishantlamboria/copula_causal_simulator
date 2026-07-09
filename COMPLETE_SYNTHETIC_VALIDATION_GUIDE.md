# Complete synthetic-validation patch

This patch completes the synthetic validation for the current maximum-indegree-two simulator.

## Included validation layers

1. Existing 1,140-run pair-copula family and parameter benchmark.
2. Validation after heterogeneous continuous marginals and empirical rank transformation.
3. Known three-variable simplified D-vines with family recovery and held-out parent-order selection.
4. End-to-end calibration and topological regeneration of a known five-node DAG.
5. Controlled violation of the simplifying assumption, comparing a global conditional copula with cross-validated quantile bins.
6. PNG/PDF plots, CSV tables, JSON summaries, and Markdown reports.

## Installation

Extract the patch into the project root while preserving folders.

Run the compact suite and create all reports:

```powershell
powershell -ExecutionPolicy Bypass -File `
  scripts\synthetic_validation\run_all_synthetic_validation.ps1
```

The compact suite is the fast verified configuration supplied with the patch.

Run the larger final configuration:

```powershell
powershell -ExecutionPolicy Bypass -File `
  scripts\synthetic_validation\run_all_synthetic_validation.ps1 `
  -Full
```

Skip tests when they have already passed:

```powershell
powershell -ExecutionPolicy Bypass -File `
  scripts\synthetic_validation\run_all_synthetic_validation.ps1 `
  -SkipTests
```

## Manual commands

```powershell
python -m pytest -q

python scripts\synthetic_validation\run_complete_synthetic_validation.py `
  --config configs\synthetic_validation\complete_synthetic_validation_compact.yaml `
  --overwrite

python scripts\synthetic_validation\plot_complete_synthetic_validation.py `
  --results-directory outputs\synthetic_validation\complete_compact `
  --report-directory reports\synthetic_validation\complete_compact
```

## Main outputs

```text
reports/synthetic_validation/pair_family_recovery/README.md
reports/synthetic_validation/complete_compact/README.md
reports/synthetic_validation/complete_compact/figures/
outputs/synthetic_validation/complete_compact/synthetic_validation_summary.json
```

## Verified status

The supplied patch was tested with Python 3.11 and pyvinecopulib 0.7.6.
All 156 tests passed. The compact suite completed successfully.

The compact results are report-ready but use few repetitions for the D-vine, full-DAG, and non-simplified experiments. Use the full configuration when final uncertainty estimates are required.
