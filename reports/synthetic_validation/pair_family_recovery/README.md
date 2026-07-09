# Synthetic pair-copula recovery report

## Experiment coverage

The completed benchmark contains **1140 successful runs** over the six candidate families, three sample sizes, three nonzero dependence strengths, and two Student-t degrees-of-freedom settings. Models were selected by BIC from the configured candidate set.

## Headline findings

- Overall exact family recovery was **78.9%**.
- At \(n=1000\), exact family recovery was **86.1%**.
- Mean absolute Kendall-tau error was **0.0140** overall and **0.0122** at \(n=1000\).
- Student-t recovery was **83.3%** for true \(df=4\), but only **32.2%** for true \(df=10\). This is consistent with the Student copula approaching the Gaussian copula as degrees of freedom increase and with BIC penalizing the extra Student parameter.
- Across all Student scenarios, **22.2%** of runs were selected as Gaussian. Exact family identification is therefore harder than recovery of dependence strength.

## Recovery by family

| True family | Runs | Exact recovery | Mean tau MAE |
| --- | --- | --- | --- |
| Independence | 60 | 96.7% | 0.0025 |
| Gaussian | 180 | 82.2% | 0.0135 |
| Student t | 360 | 57.8% | 0.0161 |
| Clayton | 180 | 90.6% | 0.0123 |
| Gumbel | 180 | 88.9% | 0.0143 |
| Frank | 180 | 90.6% | 0.0153 |

## Recovery by family and sample size

| True family | n | Runs | Exact recovery | Mean tau MAE | Mean held-out loglik gap |
| --- | --- | --- | --- | --- | --- |
| Independence | 250 | 20 | 100.0% | 0.0000 | 0.0000 |
| Independence | 500 | 20 | 100.0% | 0.0000 | 0.0000 |
| Independence | 1000 | 20 | 90.0% | 0.0075 | -0.0010 |
| Gaussian | 250 | 60 | 76.7% | 0.0146 | -0.0022 |
| Gaussian | 500 | 60 | 76.7% | 0.0144 | -0.0035 |
| Gaussian | 1000 | 60 | 93.3% | 0.0115 | -0.0025 |
| Student t | 250 | 120 | 40.8% | 0.0169 | -0.0141 |
| Student t | 500 | 120 | 63.3% | 0.0176 | -0.0077 |
| Student t | 1000 | 120 | 69.2% | 0.0138 | -0.0049 |
| Clayton | 250 | 60 | 90.0% | 0.0137 | -0.0005 |
| Clayton | 500 | 60 | 86.7% | 0.0134 | -0.0016 |
| Clayton | 1000 | 60 | 95.0% | 0.0099 | -0.0016 |
| Gumbel | 250 | 60 | 85.0% | 0.0165 | -0.0018 |
| Gumbel | 500 | 60 | 86.7% | 0.0144 | -0.0012 |
| Gumbel | 1000 | 60 | 95.0% | 0.0120 | -0.0020 |
| Frank | 250 | 60 | 81.7% | 0.0177 | -0.0017 |
| Frank | 500 | 60 | 96.7% | 0.0145 | -0.0023 |
| Frank | 1000 | 60 | 93.3% | 0.0136 | -0.0022 |

The held-out log-likelihood gap is the fitted mean test log-likelihood minus the oracle mean test log-likelihood. Values close to zero indicate that the selected model is predictively close to the known generating copula even when the exact family is not recovered.

## Representative copula figures

| Run | True family | Selected family | Exact | Tau absolute error |
| --- | --- | --- | --- | --- |
| family_gaussian__tau_0p5__n_1000__seed_6 | Gaussian | Gaussian | yes | 0.0057 |
| family_clayton__tau_0p5__n_1000__seed_19 | Clayton | Clayton | yes | 0.0053 |
| family_gumbel__tau_0p5__n_1000__seed_2 | Gumbel | Gumbel | yes | 0.0082 |
| family_frank__tau_0p2__n_250__seed_5 | Frank | Gumbel | no | 0.0391 |
| family_student__tau_0p5__df_4__n_1000__seed_17 | Student t | Student t | yes | 0.0106 |
| family_student__tau_0p5__df_10__n_1000__seed_7 | Student t | Gaussian | no | 0.0106 |

Each representative figure contains true and fitted copula-density contours, an absolute density-difference heatmap, and pseudo-observations from both models. For visual stability, density displays are capped at their joint 99.5th percentile; model evaluation itself is not clipped.

## Interpretation

The experiment validates the bivariate calibration layer: dependence strength is recovered accurately, and the main one-parameter families are usually identified. The principal limitation is exact discrimination between Student-t and nearby symmetric alternatives, especially for \(df=10\). That limitation should be reported as a model-selection identifiability issue rather than as a complete failure of distributional recovery.

This benchmark does **not** yet validate empirical marginal estimation, two-parent conditional D-vines, vine-order selection, or full-DAG generation. Those remain separate synthetic-validation milestones.
