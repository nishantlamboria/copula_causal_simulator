# Baseline: v0.4-indegree2-baseline

Captured at Git commit `3054cad59b18b8d390c9c9bee79be1024d649778` on branch `master`.

## Scientific scope

This baseline contains the current continuous-data simulator with:

- empirical marginal models and pseudo-observations;
- pair-copula fitting for one-parent mechanisms;
- candidate families: independence, Gaussian, Student t, Clayton, Gumbel, and Frank;
- BIC-based family selection with rotations enabled;
- explicit three-variable D-vines for two-parent mechanisms;
- fixed two-parent vine order `parent_1 -- parent_2 -- child`;
- graph-conditioned sampling in topological order;
- random DAG generation with maximum indegree two;
- marginal, rank-dependence, tail, and graph-aware evaluation;
- reproducibility and h-function/inverse-h-function tests.

## Known limitations

- continuous variables only;
- maximum supported indegree is two;
- fixed vine order for two-parent mechanisms;
- simplifying assumption for the conditional copula;
- no synthetic ground-truth recovery benchmark yet;
- no copula-native higher-order visual diagnostics yet;
- observational realism is evaluated, but interventional validity is not established.

## Verification

Test status: **PASSED**

The directory also contains:

- `manifest.json`: Git and environment metadata;
- `environment.txt`: Python and package versions;
- `test_output.txt`: captured pytest output;
- `file_hashes.csv`: SHA-256 hashes for source, tests, configs, data, and final tables.

## Development rule

Do not continue feature development on this tag. Create a new branch, for example:

```bash
git switch -c feature/synthetic-validation
```
