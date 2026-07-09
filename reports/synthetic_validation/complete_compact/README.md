# Complete synthetic validation

This report validates the simulator against known data-generating mechanisms.
The compact suite is a fast, reproducible verification run; the included full
configuration increases seeds and sample-size settings for final reporting.

## 1. Bivariate pair-copula calibration

- Successful runs: **1140**
- Exact family recovery: **78.9%**
- Mean absolute Kendall-tau error: **0.0140**

The main residual ambiguity is between high-degree-of-freedom Student copulas
and Gaussian copulas. Dependence strength is nevertheless recovered accurately.

## 2. Empirical marginals and inverse transformation

- Runs: **45**
- Family recovery: **100.0%**
- Mean absolute Kendall-tau error: **0.0075**
- Mean normalized marginal quantile error: **0.0277**
- Maximum rank-invariance discrepancy: **0.00e+00**

Strictly monotone marginal transformations preserve ranks exactly. The experiment
therefore confirms that the empirical pseudo-observation layer does not alter
the copula information in continuous data, while empirical inverse marginals
retain small finite-sample quantile error.

## 3. Known simplified D-vines and flexible parent order

- Runs: **15**
- Generating order selected: **100.0%**
- First pair-family recovery: **100.0%**
- Second pair-family recovery: **100.0%**
- Conditional family recovery: **100.0%**
- Mean three-dimensional Kendall error: **0.0282**
- Mean conditional log-likelihood gap from oracle: **-0.0052**

The parent order is selected using held-out conditional log-likelihood rather
than arbitrary input order. In three dimensions, order should be interpreted as
a predictive representation choice rather than a universally identifiable
causal object.

## 4. Complete known DAG

The validation DAG is

`X1 → X2`, `X1 → X3`, `X2 → X3`, `X3 → X4`, `X2 → X5`.

It contains a root, one-parent mechanisms, a two-parent mechanism, and multiple
topological generations.

- Runs: **5**
- Two-parent generating order selected: **100.0%**
- One-parent family recovery: **100.0%**
- Mean pairwise Kendall error: **0.0553**
- Mean five-dimensional energy distance: **0.0013**
- Mean marginal quantile error: **0.0854**

This is the central end-to-end check: data are generated from a known DAG,
calibrated from observations, and regenerated using fitted local mechanisms.

## 5. Violation of the simplifying assumption

The true conditional Gaussian-copula parameter varies smoothly with the
conditioning quantile. A single conditional copula is compared with a
cross-validated equal-frequency binned model.

- Runs: **5**
- Mean selected number of bins: **4.60**
- Mean held-out log-likelihood improvement: **0.0836**
- Simplified conditional-tau RMSE: **0.1884**
- Adaptive binned conditional-tau RMSE: **0.0586**

The binned model is a useful data-driven baseline and diagnostic. It reduces
conditional-tau error and improves held-out likelihood in this controlled
non-simplified setting. For higher-dimensional conditioning sets, smooth
varying-parameter models or recursive partitioning should replace naive
multidimensional binning.

## Figures

- `figures/marginal_validation_summary.*`
- `figures/dvine_validation_summary.*`
- `figures/full_dag_validation_summary.*`
- `figures/non_simplified_tau_curve.*`
- `figures/non_simplified_metrics.*`

## Scope and limitations

The synthetic suite establishes recovery under mechanisms that belong to, or
are deliberately close to, the implemented model class. It does not by itself
establish causal validity on observational real-world data. The full
configuration should be run for final uncertainty estimates; the compact run
primarily verifies correctness and produces immediate report-ready evidence.
