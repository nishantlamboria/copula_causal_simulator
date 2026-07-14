# Maximum-indegree-three implementation

## Scope

The implementation supports DAG nodes with up to three parents.  For a local
mechanism with parents `{P1,P2,P3}` and child `X`, it evaluates all six D-vine
orders

```text
Pπ(1) -- Pπ(2) -- Pπ(3) -- X
```

using one deterministic train/validation split.  The candidate with the
largest mean held-out conditional log-likelihood is selected and refitted on
all observations.

Each four-dimensional D-vine contains six pair-copulas:

```text
Tree 1: C12, C23, C3X
Tree 2: C13|2, C2X|3
Tree 3: C1X|23
```

The parent set in the DAG is never changed; only its statistical D-vine order
is selected.

## Apply the patch

Extract the patch into the repository root while preserving paths, then run:

```powershell
python -m pytest -q
```

## Compact synthetic validation

```powershell
python scripts\synthetic_validation\run_indegree3_validation.py `
  --n 1500 `
  --seed 42
```

Outputs are written to:

```text
reports/synthetic_validation/indegree_three_compact/
```

The validation checks that all six orders are scored, conditional generation
is reproducible and finite, and the generated four-dimensional dependence is
close to held-out data.

## Smoke-test a real-data library

Because the complete library contains one record for every child and every
three-parent subset, begin with a small limit:

```powershell
python scripts\build_conditional_vine_library.py `
  --config configs\datasets\sachs.yaml `
  --parent-set-size 3 `
  --limit 5
```

The outputs are:

```text
artifacts/libraries/<dataset>/conditional_vine_library_indegree3.pkl
artifacts/libraries/<dataset>/conditional_vines_indegree3/
reports/tables/<dataset>/conditional_vine_summary_indegree3.csv
```

## Rebuild all three datasets

```powershell
powershell -ExecutionPolicy Bypass -File `
  scripts\rebuild_indegree3_libraries.ps1
```

For a five-mechanism smoke test per dataset:

```powershell
powershell -ExecutionPolicy Bypass -File `
  scripts\rebuild_indegree3_libraries.ps1 `
  -SkipTests `
  -Limit 5
```

## Generate from an indegree-three graph

The coupling library, indegree-two library, and indegree-three library must
already exist.

```powershell
python scripts\generate_from_graph_indegree3.py `
  --config configs\datasets\sachs.yaml `
  --graph outputs\graphs\graph_0000_adjacency.csv `
  --n-samples 1000 `
  --seed 42 `
  --output-dir outputs\generated_indegree3\sachs
```

## Full benchmark

```powershell
python scripts\run_indegree3_benchmark.py `
  --config configs\datasets\sachs.yaml `
  --num-graphs 3 `
  --n-samples 1000 `
  --seed 42
```

## Important limitation

The adaptive quantile-binned non-simplified conditional copula remains limited
to indegree two.  The indegree-three mechanism uses a simplified
four-dimensional D-vine.  Extending adaptive binning to multiple conditioning
variables would create sparse multidimensional cells and requires a different
method, such as smooth varying-parameter copulas or recursive partitioning.

## Computational size

For `d` variables the complete indegree-three library contains

```text
d * choose(d - 1, 3)
```

local mechanisms, and each mechanism evaluates six orders.  The `--limit`
option is therefore strongly recommended for initial validation.
