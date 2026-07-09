# Flexible parent ordering for two-parent D-vines

This patch replaces the fixed local order

```text
parent_1 -- parent_2 -- child
```

with data-driven comparison of both admissible orders:

```text
parent_1 -- parent_2 -- child
parent_2 -- parent_1 -- child
```

The DAG and parent set are unchanged. Only the statistical D-vine representation used for the conditional mechanism is selected.

## Selection procedure

For each child and unordered two-parent set:

1. create one deterministic training/validation split;
2. fit both candidate orders on the same training rows;
3. score both on the same validation rows using mean conditional copula log-likelihood;
4. choose the higher-scoring order;
5. refit the selected order on all observations;
6. save only the selected full-data pair-copula components.

For order `A -- B -- X`, the validation score is

```text
mean[ log c_BX(u_B,u_X) + log c_AX|B(u_A|B,u_X|B) ].
```

The parent density factor `c_AB` is not part of the conditional score. If there are too few validation observations, the implementation compares negative full-sample BIC, so larger scores remain better.

## Apply the patch

Extract the ZIP in the repository root and preserve the directory structure.

Run:

```powershell
python -m pytest -q
```

Expected result for the supplied project snapshot:

```text
156 passed
```

## Rebuild a conditional library

Example for Sachs:

```powershell
python scripts\build_conditional_vine_library.py `
  --config configs\datasets\sachs.yaml `
  --order-strategy heldout_conditional_loglik `
  --order-validation-fraction 0.20 `
  --order-selection-seed 42 `
  --minimum-validation-rows 50
```

Repeat with:

```text
configs/datasets/diabetes.yaml
configs/datasets/breast_cancer.yaml
```

The same settings are now stored in all three dataset YAML files, so this shorter command is equivalent:

```powershell
python scripts\build_conditional_vine_library.py `
  --config configs\datasets\sachs.yaml
```

## Clean rebuild recommendation

Archive or remove the old fixed-order files before rebuilding. For Sachs, the relevant locations are configured as:

```text
artifacts/libraries/sachs/conditional_vines_indegree2/
artifacts/libraries/sachs/conditional_vine_library_indegree2.pkl
reports/tables/sachs/conditional_vine_summary_indegree2.csv
```

Use the analogous dataset directories for Diabetes and Breast Cancer.

## New summary fields

The generated CSV and pickle records now include:

```text
parent_set
selected_order
order_strategy_requested
order_selection_method
order_selection_metric
candidate_order_scores
candidate_order_statuses
candidate_order_errors
selected_order_score
alternative_order_score
order_score_margin
order_training_size
order_validation_size
```

`parent_1` and `parent_2` remain in the metadata and now denote the selected ordered roles used by the generator.

## Compatibility

Older metadata without `selected_order` or `parent_set` remains loadable. The generator reconstructs the old order from:

```text
[parent_1, parent_2, child]
```

## Commit

After tests and the three real-data rebuilds succeed:

```powershell
git add `
  src\copula_causal_sim\copulas\conditional_vine_library.py `
  src\copula_causal_sim\generators\indegree_two.py `
  scripts\build_conditional_vine_library.py `
  configs\datasets `
  tests\test_conditional_dvine.py `
  tests\test_indegree_two_generator.py `
  tests\test_flexible_vine_order.py `
  docs\FLEXIBLE_VINE_ORDER_GUIDE.md

git commit -m "Add data-driven two-parent vine ordering"
git push -u origin feature/flexible-vine-order
```
