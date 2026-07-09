# Pair-copula recovery reporting patch

This patch adds a tested reporting pipeline for the completed synthetic
pair-copula benchmark.

## Added files

- `src/copula_causal_sim/synthetic/pair_recovery_reporting.py`
- `scripts/synthetic_validation/plot_pair_recovery_results.py`
- `tests/synthetic_validation/test_pair_recovery_reporting.py`

The generated report is under:

- `reports/synthetic_validation/pair_family_recovery/README.md`
- `reports/synthetic_validation/pair_family_recovery/figures/`
- `reports/synthetic_validation/pair_family_recovery/tables/`

## Run the tests

From the repository root:

```powershell
python -m pytest -q
```

## Generate the report

When the full results remain in their normal project location:

```powershell
python scripts\synthetic_validation\plot_pair_recovery_results.py `
  --results outputs\synthetic_validation\pair_family_recovery_full\pair_family_recovery_results.csv
```

To replace or redirect the report directory:

```powershell
python scripts\synthetic_validation\plot_pair_recovery_results.py `
  --results outputs\synthetic_validation\pair_family_recovery_full\pair_family_recovery_results.csv `
  --output-directory reports\synthetic_validation\pair_family_recovery
```

The script writes every figure as both PNG and PDF. It regenerates the
representative pseudo-observations deterministically from the recorded true
model, sample size, and seed. It reconstructs the fitted copula from the
recorded selected family, rotation, and parameter vector.

## Generated scientific outputs

The report includes:

1. family-recovery confusion matrices in counts and row proportions;
2. exact family recovery versus sample size;
3. recovery stratified by true Kendall's tau;
4. estimated versus true Kendall's tau;
5. Kendall-tau absolute error versus sample size;
6. Student-t family recovery by degrees of freedom;
7. Student-t degrees-of-freedom estimates;
8. six representative true-versus-fitted copula figures containing density
   contours, absolute density differences, and pseudo-observations;
9. CSV summary tables and a Markdown interpretation report.

The copula-density displays are capped at the joint 99.5th percentile only
for visualization. The fitted models and reported likelihoods are not
clipped.
