from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CONFIGS = [
    PROJECT_ROOT / "configs" / "datasets" / "sachs.yaml",
    PROJECT_ROOT / "configs" / "datasets" / "diabetes.yaml",
]


def main() -> None:
    for config_path in CONFIGS:
        print("=" * 80)
        print(f"Building library for config: {config_path}")
        print("=" * 80)

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_coupling_library.py"),
                "--config",
                str(config_path),
            ],
            check=True,
        )

    print("\nAll coupling libraries built successfully.")


if __name__ == "__main__":
    main()