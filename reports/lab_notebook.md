# Lab Notebook

## Entry 001, Project setup and Sachs dataset decision

### Goal
Set up a clean project structure for the copula-guided causal simulator project and define the first concrete dataset task.

### Project phase
Phase 0: repository setup and project hygiene.

### Dataset
The first dataset is the Sachs protein-signaling dataset stored at:

`data/sachs.data.txt`

The file contains 11 continuous numerical variables:

- Raf
- Mek
- Plcg
- PIP2
- PIP3
- Erk
- Akt
- PKA
- PKC
- P38
- Jnk

### Decision
For the first implementation, this dataset will be used as a continuous calibration dataset. It will not yet be treated as a causal graph dataset.

### Current objective
Start T4-Sachs: build a graph-agnostic coupling library from the Sachs dataset.

### Immediate next steps
1. Load and validate the Sachs data.
2. Build empirical marginal models.
3. Transform variables to pseudo-observations in `(0, 1)`.
4. Save the first marginal artifact.
5. Then fit pairwise copulas for all variable pairs.

## Entry 002, T4b Sachs marginal handling

### Goal
Build the first marginal-handling artifact for the Sachs dataset.

### Project phase
T4b: Marginal handling.

### What I implemented
Created code to:
1. load and validate the Sachs dataset,
2. summarize all variables,
3. transform all variables to pseudo-observations in `(0, 1)` using average ranks,
4. fit empirical marginal models,
5. save the marginal library as a pickle artifact.

### Output artifacts
- `artifacts/sachs_pseudo_observations.csv`
- `artifacts/sachs_marginals_v1.pkl`
- `reports/tables/sachs_dataset_summary.csv`
- `reports/tables/sachs_reconstructed_summary.csv`

### Decision
The Sachs dataset is currently treated as a continuous calibration dataset. No causal graph is used at this stage.

### Next step
Implement T4c: fit pairwise copulas for all 55 variable pairs and save a Sachs coupling library.