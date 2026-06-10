# Copula Causal Simulator

This project implements a copula-calibrated synthetic data generator for supervised causal learning experiments.

The current implementation builds dataset-specific coupling libraries from continuous tabular datasets. Each coupling library contains empirical marginal models, pseudo-observations, and fitted pairwise copula models.

## Current status

The project currently supports config-driven coupling-library construction for continuous tabular datasets.

Implemented datasets:

- Sachs protein-signaling dataset
- Scikit-learn diabetes dataset

## Main command

```powershell
python scripts/build_coupling_library.py --config configs/datasets/sachs.yaml