# Compact indegree-three synthetic validation

A known four-dimensional D-vine was simulated with generating order
`P1 -- P2 -- P3 -- X`. The production calibrator evaluated all six parent
permutations and selected `P1--P2--P3--X` using mean
held-out conditional log-likelihood.

- All six orders scored: **True**
- Score margin over second-best order: **0.095778**
- Held-out pairwise Kendall MAE: **0.009184**
- Held-out four-dimensional energy distance: **0.000482**
- Generated values finite and inside `(0,1)`: **True**

This validation checks order scoring, the complete six-copula local mechanism,
and conditional generation for a node with three parents. It does not extend
the adaptive non-simplified binning method beyond indegree two.
