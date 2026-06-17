# Copula Causal Simulator

This project implements a copula-calibrated synthetic data generator for supervised causal learning experiments.

The system builds coupling libraries from continuous tabular datasets and uses them to generate graph-conditioned synthetic data for random or user-specified DAGs.

## Current functionality

- Config-driven dataset loading
- Empirical marginal modeling
- Pseudo-observation transformation using ranks
- Pairwise copula fitting with AIC/BIC selection
- Conditional 3-variable vine fitting for indegree-2 mechanisms
- Random DAG generation with bounded indegree
- Graph-conditioned generation for max indegree 1
- Graph-conditioned generation for max indegree 2
- Realism and graph-aware evaluation metrics
- Reproducible benchmark runner for indegree 1

## Install

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

Build a coupling library
python scripts/build_coupling_library.py --config configs/datasets/sachs.yaml
Build conditional vine mechanisms
python scripts/build_conditional_vine_library.py --config configs/datasets/sachs.yaml --parent-set-size 2
Generate random graphs
python scripts/generate_random_graphs.py --num-graphs 1 --num-nodes 11 --edge-prob 0.30 --max-indegree 2 --seed 42
Generate data from an indegree-2 graph
python scripts/generate_from_graph_indegree2.py --config configs/datasets/sachs.yaml --graph artifacts/graphs/sachs_indegree2/graph_0000_adjacency.csv --n-samples 1000 --seed 42
Evaluate generated data
python scripts/evaluate_generated_data.py --config configs/datasets/sachs.yaml --generated-x outp