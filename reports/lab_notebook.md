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