# Copula Sachs Generator

This project implements a first prototype of a copula-calibrated synthetic data generator using the Sachs protein-signaling dataset.

The current focus is not yet graph-conditioned causal generation. The first milestone is to build a graph-agnostic coupling library from one continuous dataset.

## Current phase

T4-Sachs: Build a coupling library from the Sachs dataset.

## Planned stages

1. Dataset loading and validation
2. Empirical marginal handling
3. Pseudo-observation transformation
4. Pairwise copula fitting
5. Coupling library artifact
6. Baseline generators
7. Evaluation
8. Graph-conditioned generation

## Project structure

- `data/`: raw input data
- `src/copula_sachs/`: reusable Python code
- `scripts/`: runnable scripts
- `artifacts/`: generated model/library artifacts
- `reports/`: lab notebook, tables, and figures
- `tests/`: unit tests