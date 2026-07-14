# Copula Causal Simulator

This repository implements a copula-calibrated synthetic-data generator for supervised causal-learning experiments.

Given a continuous tabular dataset and a directed acyclic graph (DAG), the simulator:

1. fits empirical marginal models;
2. transforms observations to pseudo-observations;
3. fits pair-copula and conditional vine mechanisms;
4. stores reusable mechanism libraries; and
5. generates graph-conditioned synthetic observations.

The current implementation supports DAGs with maximum indegree up to three.

## Main features

- Config-driven loading of continuous tabular datasets
- Empirical marginal estimation
- Rank-based pseudo-observation transformation
- Pair-copula selection using AIC, BIC, or mBIC
- Conditional three-variable D-vines for indegree-two mechanisms
- Conditional four-variable D-vines for indegree-three mechanisms
- Selection among all six candidate parent orders for three-parent mechanisms
- Optional adaptive conditional-copula models
- Random DAG generation with a bounded maximum indegree
- Graph-conditioned generation for maximum indegree 1, 2, or 3
- Resumable parallel fitting of exhaustive indegree-three libraries
- Synthetic validation for order selection and distributional fidelity
- Saved precomputed indegree-three artifacts distributed through Git LFS
- Unit and integration tests

## Supported datasets

The repository includes configurations for:

- Sachs
- Diabetes
- Breast Cancer

Configuration files are stored under:

```text
configs/datasets/
```

The standard dataset configurations define the raw-data path, fitting options, artifact directory, and report directory.

## Installation

Python 3.11 is recommended.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
```

Set the source directory on `PYTHONPATH` when running scripts directly:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

### Linux or macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

## Running the tests

Run the full test suite:

```powershell
python -m pytest -q
```

Run only the indegree-three tests:

```powershell
python -m pytest tests\test_indegree_three.py -q
```

On Linux or macOS, replace backslashes with forward slashes.

## Repository structure

```text
copula_causal_simulator/
├── artifacts/                  # Fitted libraries and packaged artifacts
├── configs/
│   └── datasets/               # Dataset-specific YAML configurations
├── data/                       # Input datasets
├── graphs/                     # Saved adjacency matrices
├── reports/                    # Tables, figures, and validation outputs
├── scripts/                    # Build, generation, evaluation, and validation CLIs
├── src/
│   └── copula_causal_sim/      # Python package
├── tests/                      # Automated tests
├── requirements.txt
└── README.md
```

## Graph convention

Graphs are supplied as headerless square CSV adjacency matrices.

The convention is:

```text
adj[i, j] = 1  means  variable i -> variable j
```

The variable order must match the column order of the corresponding dataset.

The graph must:

- be square;
- contain no self-loops;
- be acyclic; and
- respect the maximum indegree supported by the selected generator.

## Standard library-building workflow

The lower-indegree workflow consists of marginal, pair-copula, and conditional-vine libraries.

### Build marginal and coupling libraries

Use the repository's library-building scripts for the dataset configuration of interest. Check the exact command-line options available in the current branch before running:

```powershell
python scripts\build_all_libraries.py --help
python scripts\build_coupling_library.py --help
```

The dataset configuration determines the raw-data location and output directories.

### Build indegree-two conditional mechanisms

```powershell
python scripts\build_conditional_vine_library.py `
    --config configs\datasets\sachs.yaml `
    --parent-set-size 2
```

### Build indegree-three conditional mechanisms serially

```powershell
python scripts\build_conditional_vine_library.py `
    --config configs\datasets\sachs.yaml `
    --parent-set-size 3
```

For complete indegree-three libraries, the parallel runner is recommended instead.

## Exhaustive indegree-three fitting

For a dataset with \(d\) variables, the number of child/three-parent mechanisms is

```text
d * choose(d - 1, 3)
```

The complete libraries contain:

| Dataset | Variables | Indegree-three mechanisms |
|---|---:|---:|
| Sachs | 11 | 1,320 |
| Diabetes | 10 | 840 |
| Breast Cancer | 12 | 1,980 |
| **Total** |  | **4,140** |

### Prevent nested thread oversubscription

Before launching multiprocessing on Windows:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
```

### Run the exhaustive parallel fitter

Example for Sachs with eight worker processes:

```powershell
python -u scripts\run_indegree3_parallel.py `
    --config configs\datasets\sachs.yaml `
    --workers 8 `
    --checkpoint-every 1 `
    --restart
```

Run the other datasets sequentially:

```powershell
python -u scripts\run_indegree3_parallel.py `
    --config configs\datasets\diabetes.yaml `
    --workers 8 `
    --checkpoint-every 1 `
    --restart
```

```powershell
python -u scripts\run_indegree3_parallel.py `
    --config configs\datasets\breast_cancer.yaml `
    --workers 8 `
    --checkpoint-every 1 `
    --restart
```

Do not run all three commands simultaneously unless the machine has sufficient CPU and memory capacity.

### Resume an interrupted run

Rerun the same command without `--restart`:

```powershell
python -u scripts\run_indegree3_parallel.py `
    --config configs\datasets\sachs.yaml `
    --workers 8 `
    --checkpoint-every 1
```

The runner loads the checkpoint and fits only missing mechanisms.

### Retry failed mechanisms

```powershell
python -u scripts\run_indegree3_parallel.py `
    --config configs\datasets\sachs.yaml `
    --workers 8 `
    --retry-failed
```

### Small benchmark or smoke test

```powershell
python -u scripts\run_indegree3_parallel.py `
    --config configs\datasets\sachs.yaml `
    --workers 8 `
    --limit 40 `
    --restart
```

## Precomputed indegree-three artifacts

The exhaustive indegree-three libraries are packaged under:

```text
artifacts/precomputed_indegree3/
```

The package contains one archive and one manifest per dataset, together with SHA-256 checksums.

The archives are tracked through Git LFS. After cloning or pulling the repository, run:

```powershell
git lfs install
git lfs pull
git lfs ls-files
```

The expected archives are:

```text
sachs_indegree3.tar.gz
diabetes_indegree3.tar.gz
breast_cancer_indegree3.tar.gz
```

The standard dataset configurations use artifact directories of the form:

```text
artifacts/libraries/<dataset>/
```

Extract each archive into the corresponding configured artifact directory. For example:

```powershell
New-Item -ItemType Directory `
    -Force `
    -Path artifacts\libraries\sachs |
    Out-Null

tar -xzf `
    artifacts\precomputed_indegree3\sachs_indegree3.tar.gz `
    -C artifacts\libraries\sachs
```

Repeat for Diabetes and Breast Cancer.

Each archive contains the fitted indegree-three library and its JSON mechanism models. The included manifest records the expected mechanism count and SHA-256 checksum.

## Packaging newly fitted indegree-three artifacts

After rebuilding all three complete libraries:

```powershell
python -u scripts\package_indegree3_artifacts.py
```

The script validates the result counts, creates compressed archives, writes per-dataset manifests, and generates `SHA256SUMS`.

Large archives should remain tracked with Git LFS rather than normal Git objects.

## Generating random graphs

Example with 11 nodes and maximum indegree three:

```powershell
python scripts\generate_random_graphs.py `
    --num-graphs 1 `
    --num-nodes 11 `
    --edge-prob 0.30 `
    --max-indegree 3 `
    --seed 42
```

## Generating synthetic data

### Maximum indegree one

```powershell
python scripts\generate_from_graph_indegree1.py `
    --config configs\datasets\sachs.yaml `
    --graph path\to\adjacency.csv `
    --n-samples 1000 `
    --seed 42
```

### Maximum indegree two

```powershell
python scripts\generate_from_graph_indegree2.py `
    --config configs\datasets\sachs.yaml `
    --graph path\to\adjacency.csv `
    --n-samples 1000 `
    --seed 42
```

### Maximum indegree three

```powershell
python scripts\generate_from_graph_indegree3.py `
    --config configs\datasets\sachs.yaml `
    --graph graphs\end_to_end\sachs_mixed_indegree.csv `
    --n-samples 2000 `
    --seed 42 `
    --output-dir outputs\end_to_end\sachs_seed42
```

The indegree-three generator writes:

```text
generated_u.csv
generated_x.csv
generation_metadata.json
```

`generated_u.csv` contains pseudo-observations in the unit interval.  
`generated_x.csv` contains observations transformed back to the original marginal scales.

## Evaluating generated data

The repository includes `scripts/evaluate_generated_data.py` for distributional and graph-aware diagnostics. Inspect the exact interface in the current branch before running:

```powershell
python scripts\evaluate_generated_data.py --help
```

## Indegree-three synthetic validation

A compact validation run is available through:

```powershell
python -u scripts\synthetic_validation\run_indegree3_validation.py `
    --n 500 `
    --seed 42 `
    --output-dir reports\synthetic_validation\indegree_three_compact
```

The validation:

- generates a controlled four-variable mechanism;
- evaluates all six candidate parent orders;
- records the selected order and score margin;
- computes pairwise Kendall-\(\tau\) mean absolute error;
- computes four-dimensional energy distance; and
- verifies that generated values are finite and lie in \([0,1]\).

The validation split requires a sufficiently large sample. Final multi-seed experiments use:

```text
n in {500, 1000, 2000}
seeds in {1, 2, 3, 4, 5}
15 runs in total
```

Aggregated outputs are stored under:

```text
reports/final_validation/indegree3_multiseed/
```

## Final release validation

After building the libraries, generating the three end-to-end datasets, and aggregating the 15 validation runs, execute:

```powershell
python -u scripts\validate_final_indegree3_release.py
```

The release check verifies:

- 1,320 successful Sachs mechanisms;
- 840 successful Diabetes mechanisms;
- 1,980 successful Breast Cancer mechanisms;
- all three saved indegree-three libraries;
- valid end-to-end generation for all three datasets;
- 15 complete multi-seed validation runs;
- finite generated values;
- unit-interval validity; and
- scoring of all six candidate orders.

A successful run ends with:

```text
All final indegree-three technical checks passed.
```

## Reproducibility notes

- Use the dataset YAML files to preserve fitting options and output paths.
- Record the random seed for every graph-generation and data-generation run.
- Keep `pyvinecopulib`, Python, and dependency versions fixed through the environment.
- Use one numerical-library thread per multiprocessing worker.
- Do not use `--restart` when resuming an interrupted exhaustive build.
- The fitted D-vine has an ordered internal representation, while library lookup is based on the unordered parent set.
- Precomputed archives should be distributed through Git LFS.
- Generated outputs, local smoke-test folders, logs, and checkpoints should not be committed.

## Useful commands

Show a script's options:

```powershell
python scripts\run_indegree3_parallel.py --help
python scripts\generate_from_graph_indegree3.py --help
```

Show the current commit:

```powershell
git rev-parse HEAD
```

Show Git LFS objects:

```powershell
git lfs ls-files
```

Check repository state:

```powershell
git status
```

## Citation and acknowledgement

When using this software in a publication, cite the corresponding project report or repository release.

Calculations performed on the Lichtenberg high-performance computer of TU Darmstadt should include the acknowledgement required by the computing centre:

> Calculations for this research were conducted on the Lichtenberg high performance computer of the TU Darmstadt.
