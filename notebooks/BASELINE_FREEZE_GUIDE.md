# Freeze the current indegree-two implementation

The uploaded repository already contains an older tag, `v0.3-indegree2-stable`, but the current HEAD also includes the later evaluation and presentation-benchmark work. Freeze the current state under a new annotated tag:

```text
v0.4-indegree2-baseline
```

## 1. Add the capture script

Copy `freeze_baseline.py` to:

```text
scripts/freeze_baseline.py
```

## 2. Check the working tree

From the project root:

```powershell
git status --short
git diff --ignore-space-at-eol --stat
```

Do not freeze a mixture of intentional edits and uncommitted experiments. If the only apparent differences are CRLF/LF line-ending changes, resolve those separately before proceeding.

## 3. Run the complete test suite

```powershell
python -m pytest -q
```

All tests should pass before the baseline is tagged.

## 4. Commit the exact current state

Only do this when `git diff` contains intended changes:

```powershell
git add .
git commit -m "Freeze stable indegree-two baseline"
```

If the repository is already clean, no new commit is needed.

## 5. Capture metadata and create the immutable tag

```powershell
python scripts\freeze_baseline.py `
  --name v0.4-indegree2-baseline `
  --create-tag
```

The script:

- refuses to proceed from a dirty working tree;
- runs `pytest -q`;
- captures `pip freeze` and Python/platform information;
- hashes source files, tests, configs, calibration data, and final benchmark tables;
- writes baseline documentation under `reports/baselines/v0.4-indegree2-baseline/`;
- creates the annotated Git tag only if tests pass.

## 6. Push the commit and tag

```powershell
git push origin master
git push origin v0.4-indegree2-baseline
```

If your main branch has a different name, replace `master` accordingly.

## 7. Start the next phase on a new branch

```powershell
git switch -c feature/synthetic-validation
```

All synthetic-ground-truth work should happen on that branch. The baseline tag remains an immutable reference for regression comparisons.
