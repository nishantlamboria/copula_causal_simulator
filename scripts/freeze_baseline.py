from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "reports" / "baselines"

INCLUDE_PATTERNS = (
    "src/**/*.py",
    "scripts/**/*.py",
    "tests/**/*.py",
    "configs/**/*.yaml",
    "configs/**/*.yml",
    "configs/**/*.json",
    "data/*.csv",
    "data/*.txt",
    "data/*.json",
    "reports/final_tables/*.csv",
    "README.md",
    "requirements.txt",
    ".gitignore",
)

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the current copula-simulator implementation as a "
            "reproducible baseline."
        )
    )
    parser.add_argument(
        "--name",
        default="v0.4-indegree2-baseline",
        help="Baseline and Git tag name.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Capture the baseline without running pytest.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow capture from a working tree with uncommitted changes.",
    )
    parser.add_argument(
        "--create-tag",
        action="store_true",
        help="Create an annotated Git tag after a successful capture.",
    )
    return parser.parse_args()


def run(
    command: list[str],
    *,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=check,
        text=True,
        capture_output=capture,
    )


def git_text(*args: str) -> str:
    result = run(["git", *args])
    return result.stdout.strip()


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip())
    cleaned = cleaned.strip("-.")
    if not cleaned:
        raise ValueError("Baseline name becomes empty after sanitization.")
    return cleaned


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def should_include(path: Path) -> bool:
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return False
    return not any(part in EXCLUDED_PARTS for part in relative.parts)


def iter_baseline_files() -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in INCLUDE_PATTERNS:
        for path in PROJECT_ROOT.glob(pattern):
            if not path.is_file() or not should_include(path):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def collect_hashes() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(iter_baseline_files()):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def write_hashes_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(records)


def capture_environment() -> str:
    result = run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
    )
    lines = [
        f"python_executable={sys.executable}",
        f"python_version={platform.python_version()}",
        f"platform={platform.platform()}",
        "",
        "# pip freeze",
    ]
    if result.returncode == 0:
        lines.append(result.stdout.rstrip())
    else:
        lines.append("pip freeze failed:")
        lines.append(result.stderr.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def run_tests() -> tuple[int, str]:
    result = run(
        [sys.executable, "-m", "pytest", "-q"],
        check=False,
    )
    output = result.stdout
    if result.stderr:
        output += "\n--- stderr ---\n" + result.stderr
    return result.returncode, output.rstrip() + "\n"


def make_readme(
    *,
    baseline_name: str,
    commit: str,
    branch: str,
    test_status: str,
) -> str:
    return f"""# Baseline: {baseline_name}

Captured at Git commit `{commit}` on branch `{branch}`.

## Scientific scope

This baseline contains the current continuous-data simulator with:

- empirical marginal models and pseudo-observations;
- pair-copula fitting for one-parent mechanisms;
- candidate families: independence, Gaussian, Student t, Clayton, Gumbel, and Frank;
- BIC-based family selection with rotations enabled;
- explicit three-variable D-vines for two-parent mechanisms;
- fixed two-parent vine order `parent_1 -- parent_2 -- child`;
- graph-conditioned sampling in topological order;
- random DAG generation with maximum indegree two;
- marginal, rank-dependence, tail, and graph-aware evaluation;
- reproducibility and h-function/inverse-h-function tests.

## Known limitations

- continuous variables only;
- maximum supported indegree is two;
- fixed vine order for two-parent mechanisms;
- simplifying assumption for the conditional copula;
- no synthetic ground-truth recovery benchmark yet;
- no copula-native higher-order visual diagnostics yet;
- observational realism is evaluated, but interventional validity is not established.

## Verification

Test status: **{test_status}**

The directory also contains:

- `manifest.json`: Git and environment metadata;
- `environment.txt`: Python and package versions;
- `test_output.txt`: captured pytest output;
- `file_hashes.csv`: SHA-256 hashes for source, tests, configs, data, and final tables.

## Development rule

Do not continue feature development on this tag. Create a new branch, for example:

```bash
git switch -c feature/synthetic-validation
```
"""


def tag_exists(tag_name: str) -> bool:
    result = run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag_name}"],
        check=False,
    )
    return result.returncode == 0


def create_annotated_tag(tag_name: str, commit: str) -> None:
    if tag_exists(tag_name):
        raise RuntimeError(f"Git tag already exists: {tag_name}")
    run(
        [
            "git",
            "tag",
            "-a",
            tag_name,
            commit,
            "-m",
            "Freeze stable continuous max-indegree-two baseline",
        ]
    )


def main() -> None:
    args = parse_args()
    baseline_name = sanitize_name(args.name)

    try:
        git_root = Path(git_text("rev-parse", "--show-toplevel")).resolve()
    except Exception as exc:
        raise SystemExit(f"Not inside a Git repository: {exc}") from exc

    if git_root != PROJECT_ROOT.resolve():
        raise SystemExit(
            "The script must be located at scripts/freeze_baseline.py "
            "inside the repository being frozen."
        )

    dirty_status = git_text("status", "--porcelain")
    if dirty_status and not args.allow_dirty:
        print("Working tree is not clean:\n")
        print(dirty_status)
        raise SystemExit(
            "Commit or intentionally discard changes before freezing. "
            "Use --allow-dirty only when this is deliberate."
        )

    commit = git_text("rev-parse", "HEAD")
    short_commit = git_text("rev-parse", "--short=12", "HEAD")
    branch = git_text("branch", "--show-current") or "detached-HEAD"
    tags_at_commit = git_text("tag", "--points-at", "HEAD").splitlines()

    baseline_dir = BASELINE_ROOT / baseline_name
    baseline_dir.mkdir(parents=True, exist_ok=True)

    environment_text = capture_environment()
    (baseline_dir / "environment.txt").write_text(
        environment_text,
        encoding="utf-8",
    )

    if args.skip_tests:
        test_returncode = None
        test_output = "Tests were skipped by request.\n"
        test_status = "SKIPPED"
    else:
        test_returncode, test_output = run_tests()
        test_status = "PASSED" if test_returncode == 0 else "FAILED"

    (baseline_dir / "test_output.txt").write_text(
        test_output,
        encoding="utf-8",
    )

    file_hashes = collect_hashes()
    write_hashes_csv(
        baseline_dir / "file_hashes.csv",
        file_hashes,
    )

    manifest = {
        "baseline_name": baseline_name,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": commit,
            "short_commit": short_commit,
            "branch": branch,
            "tags_at_commit_before_capture": tags_at_commit,
            "working_tree_dirty": bool(dirty_status),
            "dirty_status": dirty_status.splitlines(),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "tests": {
            "status": test_status,
            "returncode": test_returncode,
            "command": f"{sys.executable} -m pytest -q",
        },
        "files": {
            "count": len(file_hashes),
            "hash_algorithm": "sha256",
            "manifest_file": "file_hashes.csv",
        },
    }

    (baseline_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    (baseline_dir / "README.md").write_text(
        make_readme(
            baseline_name=baseline_name,
            commit=commit,
            branch=branch,
            test_status=test_status,
        ),
        encoding="utf-8",
    )

    if test_status == "FAILED":
        print(test_output)
        raise SystemExit(
            "Tests failed. Baseline metadata was captured for diagnosis, "
            "but no Git tag was created."
        )

    if args.create_tag:
        create_annotated_tag(baseline_name, commit)
        print(f"Created annotated Git tag: {baseline_name}")

    print("Baseline captured successfully.")
    print(f"Commit: {commit}")
    print(f"Tests: {test_status}")
    print(f"Output: {baseline_dir}")


if __name__ == "__main__":
    main()
