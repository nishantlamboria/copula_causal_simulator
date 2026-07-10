# Adaptive conditional-copula diagnostics

Input: `reports\tables\sachs\conditional_vine_summary_indegree2.csv`

- Successful two-parent mechanisms: **495**
- Mechanisms selecting more than one bin: **24** (4.8%)
- Mechanisms retaining the simplifying assumption: **471**
- Median raw best-score improvement over the simplified model: **0.000000**

A multi-bin model is selected only when its held-out mean conditional
log-likelihood exceeds the simplified model by the configured minimum
improvement.  Non-edge or causal interpretations should not be attached to
this diagnostic: it concerns only the statistical conditional-copula
representation for a fixed DAG parent set and selected D-vine order.
