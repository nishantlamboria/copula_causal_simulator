from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, ks_2samp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from copula_causal_sim.config import load_config
from copula_causal_sim.copulas.marginals import EmpiricalMarginalLibrary
from copula_causal_sim.generators.indegree_one import load_paircopula_metadata
from copula_causal_sim.generators.indegree_two import (
    IndegreeTwoGenerator,
    load_conditional_vine_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate conditional-vine sampling for one child | parent1, parent2 mechanism."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Dataset config path, e.g. configs/datasets/sachs.yaml",
    )

    parser.add_argument(
        "--child",
        type=str,
        default=None,
        help="Optional child variable name. If omitted, a strong fitted triple is selected automatically.",
    )

    parser.add_argument(
        "--parent1",
        type=str,
        default=None,
        help="Optional first parent variable name.",
    )

    parser.add_argument(
        "--parent2",
        type=str,
        default=None,
        help="Optional second parent variable name.",
    )

    parser.add_argument(
        "--num-repeats",
        type=int,
        default=20,
        help="Number of conditional child samples per real parent row.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/validation/conditional_vine",
    )

    return parser.parse_args()


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, tuple):
            return [sanitize(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            val = float(obj)
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize(payload), f, indent=2)


def choose_record(
    records: list[dict],
    child: str | None,
    parent1: str | None,
    parent2: str | None,
) -> dict:
    ok_records = [record for record in records if record.get("status") == "ok"]

    if not ok_records:
        raise ValueError("No successful conditional-vine records found.")

    if child is not None or parent1 is not None or parent2 is not None:
        if child is None or parent1 is None or parent2 is None:
            raise ValueError(
                "If specifying a mechanism manually, provide all three: "
                "--child, --parent1, and --parent2."
            )

        requested_parent_set = {parent1, parent2}

        for record in ok_records:
            if (
                record["child"] == child
                and {record["parent_1"], record["parent_2"]} == requested_parent_set
            ):
                return record

        raise ValueError(
            "Requested conditional-vine mechanism not found.\n"
            f"Requested: {child} | {parent1}, {parent2}"
        )

    # Automatic choice:
    # select the successfully fitted triple with strongest combined empirical
    # child-parent Kendall association.
    def score(record: dict) -> float:
        tau1 = record.get("kendall_parent_1_child")
        tau2 = record.get("kendall_parent_2_child")

        tau1 = 0.0 if tau1 is None or pd.isna(tau1) else abs(float(tau1))
        tau2 = 0.0 if tau2 is None or pd.isna(tau2) else abs(float(tau2))

        return tau1 + tau2

    return max(ok_records, key=score)


def assign_parent_bin(parent1_u: np.ndarray, parent2_u: np.ndarray) -> np.ndarray:
    """
    Split parent space into four bins using pseudo-observation threshold 0.5.

    LL: parent1 low,  parent2 low
    LH: parent1 low,  parent2 high
    HL: parent1 high, parent2 low
    HH: parent1 high, parent2 high
    """
    labels = []

    for u1, u2 in zip(parent1_u, parent2_u):
        first = "L" if u1 <= 0.5 else "H"
        second = "L" if u2 <= 0.5 else "H"
        labels.append(first + second)

    return np.asarray(labels)


def safe_corr(func, x: np.ndarray, y: np.ndarray) -> float | None:
    try:
        value = func(x, y).statistic
        if value is None or np.isnan(value):
            return None
        return float(value)
    except Exception:
        return None


def extract_ks_result(ks_result) -> tuple[float, float]:
    try:
        statistic = ks_result.statistic
        pvalue = ks_result.pvalue
    except AttributeError:
        statistic = ks_result[0]
        pvalue = ks_result[1]
    return float(statistic), float(pvalue)


def conditional_bin_diagnostics(
    real_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    child: str,
) -> pd.DataFrame:
    rows = []
    bins = ["LL", "LH", "HL", "HH"]

    for bin_label in bins:
        real_bin = real_df[real_df["parent_bin"] == bin_label]
        gen_bin = generated_df[generated_df["parent_bin"] == bin_label]

        if len(real_bin) == 0 or len(gen_bin) == 0:
            rows.append({
                "parent_bin": bin_label,
                "real_count": len(real_bin),
                "generated_count": len(gen_bin),
                "real_child_mean": None,
                "generated_child_mean": None,
                "abs_mean_error": None,
                "real_child_q25": None,
                "generated_child_q25": None,
                "abs_q25_error": None,
                "real_child_median": None,
                "generated_child_median": None,
                "abs_median_error": None,
                "real_child_q75": None,
                "generated_child_q75": None,
                "abs_q75_error": None,
                "ks_statistic": None,
                "ks_pvalue": None,
            })
            continue

        real_child = real_bin[child].to_numpy(dtype=float)
        gen_child = gen_bin[child].to_numpy(dtype=float)

        ks_statistic, ks_pvalue = extract_ks_result(ks_2samp(real_child, gen_child))

        real_q25 = float(np.quantile(real_child, 0.25))
        gen_q25 = float(np.quantile(gen_child, 0.25))

        real_med = float(np.quantile(real_child, 0.50))
        gen_med = float(np.quantile(gen_child, 0.50))

        real_q75 = float(np.quantile(real_child, 0.75))
        gen_q75 = float(np.quantile(gen_child, 0.75))

        rows.append({
            "parent_bin": bin_label,
            "real_count": int(len(real_bin)),
            "generated_count": int(len(gen_bin)),

            "real_child_mean": float(real_child.mean()),
            "generated_child_mean": float(gen_child.mean()),
            "abs_mean_error": float(abs(real_child.mean() - gen_child.mean())),

            "real_child_q25": real_q25,
            "generated_child_q25": gen_q25,
            "abs_q25_error": float(abs(real_q25 - gen_q25)),

            "real_child_median": real_med,
            "generated_child_median": gen_med,
            "abs_median_error": float(abs(real_med - gen_med)),

            "real_child_q75": real_q75,
            "generated_child_q75": gen_q75,
            "abs_q75_error": float(abs(real_q75 - gen_q75)),

            "ks_statistic": float(ks_statistic),
            "ks_pvalue": float(ks_pvalue),
        })

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    if args.num_repeats <= 0:
        raise ValueError("--num-repeats must be positive.")

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    dataset_id = config["dataset_id"]
    artifact_dir = PROJECT_ROOT / config["outputs"]["artifact_dir"]

    output_dir = (
        PROJECT_ROOT
        / args.output_dir
        / dataset_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    marginal_path = artifact_dir / "marginals.pkl"
    paircopula_path = artifact_dir / "coupling_library.pkl"
    conditional_vine_path = artifact_dir / "conditional_vine_library_indegree2.pkl"
    pseudo_path = artifact_dir / "pseudo_observations.csv"

    if not marginal_path.exists():
        raise FileNotFoundError(f"Marginal library not found: {marginal_path}")

    if not paircopula_path.exists():
        raise FileNotFoundError(f"Pair-copula metadata not found: {paircopula_path}")

    if not conditional_vine_path.exists():
        raise FileNotFoundError(
            f"Conditional vine library not found: {conditional_vine_path}\n"
            "Run scripts/build_conditional_vine_library.py first."
        )

    if not pseudo_path.exists():
        raise FileNotFoundError(
            f"Pseudo-observations not found: {pseudo_path}\n"
            "Run scripts/build_coupling_library.py first."
        )

    print(f"Loading artifacts for dataset: {dataset_id}")

    marginal_library = EmpiricalMarginalLibrary.load(marginal_path)
    paircopula_metadata = load_paircopula_metadata(paircopula_path)
    conditional_vine_metadata = load_conditional_vine_metadata(conditional_vine_path)

    u_df = pd.read_csv(pseudo_path)
    columns = list(u_df.columns)
    column_to_index = {col: idx for idx, col in enumerate(columns)}

    record = choose_record(
        records=conditional_vine_metadata["records"],
        child=args.child,
        parent1=args.parent1,
        parent2=args.parent2,
    )

    child = record["child"]
    parent1 = record["parent_1"]
    parent2 = record["parent_2"]

    print("\nSelected conditional mechanism:")
    print(f"  {child} | {parent1}, {parent2}")
    print(f"  model_file: {record['model_file']}")
    print(f"  status: {record['status']}")
    print(f"  bic: {record.get('bic')}")
    print(f"  kendall_parent_1_child: {record.get('kendall_parent_1_child')}")
    print(f"  kendall_parent_2_child: {record.get('kendall_parent_2_child')}")

    generator = IndegreeTwoGenerator(
        marginal_library=marginal_library,
        paircopula_metadata=paircopula_metadata,
        conditional_vine_metadata=conditional_vine_metadata,
        artifact_dir=artifact_dir,
        clip_eps=float(config["data"].get("clip_eps", 1e-6)),
    )

    rng = np.random.default_rng(args.seed)

    # Repeat real parent values to reduce Monte Carlo noise in the generated child.
    parent1_values = np.repeat(u_df[parent1].to_numpy(dtype=float), args.num_repeats)
    parent2_values = np.repeat(u_df[parent2].to_numpy(dtype=float), args.num_repeats)

    n_generated = len(parent1_values)

    u_matrix = np.full(
        shape=(n_generated, len(columns)),
        fill_value=0.5,
        dtype=float,
    )

    u_matrix[:, column_to_index[parent1]] = parent1_values
    u_matrix[:, column_to_index[parent2]] = parent2_values

    generated_child = generator._sample_child_given_two_parents(
        child_name=child,
        parent_names=[parent1, parent2],
        u_matrix=u_matrix,
        rng=rng,
    )

    if np.isnan(generated_child).any():
        raise RuntimeError("Generated conditional child samples contain NaN values.")

    if np.any((generated_child <= 0.0) | (generated_child >= 1.0)):
        raise RuntimeError("Generated conditional child samples are outside (0, 1).")

    real_parent1 = u_df[parent1].to_numpy(dtype=float)
    real_parent2 = u_df[parent2].to_numpy(dtype=float)
    real_child = u_df[child].to_numpy(dtype=float)

    real_bins = assign_parent_bin(real_parent1, real_parent2)
    generated_bins = assign_parent_bin(parent1_values, parent2_values)

    real_validation_df = pd.DataFrame({
        parent1: real_parent1,
        parent2: real_parent2,
        child: real_child,
        "parent_bin": real_bins,
        "source": "real",
    })

    generated_validation_df = pd.DataFrame({
        parent1: parent1_values,
        parent2: parent2_values,
        child: generated_child,
        "parent_bin": generated_bins,
        "source": "generated",
    })

    # Overall dependence diagnostics in pseudo-observation space.
    real_kendall_p1_child = safe_corr(kendalltau, real_parent1, real_child)
    real_kendall_p2_child = safe_corr(kendalltau, real_parent2, real_child)
    gen_kendall_p1_child = safe_corr(kendalltau, parent1_values, generated_child)
    gen_kendall_p2_child = safe_corr(kendalltau, parent2_values, generated_child)

    real_spearman_p1_child = safe_corr(spearmanr, real_parent1, real_child)
    real_spearman_p2_child = safe_corr(spearmanr, real_parent2, real_child)
    gen_spearman_p1_child = safe_corr(spearmanr, parent1_values, generated_child)
    gen_spearman_p2_child = safe_corr(spearmanr, parent2_values, generated_child)

    bin_diag = conditional_bin_diagnostics(
        real_df=real_validation_df,
        generated_df=generated_validation_df,
        child=child,
    )

    mean_abs_bin_mean_error = float(bin_diag["abs_mean_error"].dropna().mean())
    median_abs_bin_median_error = float(bin_diag["abs_median_error"].dropna().median())

    summary = {
        "dataset_id": dataset_id,
        "selected_mechanism": {
            "child": child,
            "parent_1": parent1,
            "parent_2": parent2,
            "model_file": record["model_file"],
            "bic": record.get("bic"),
            "aic": record.get("aic"),
            "loglik": record.get("loglik"),
            "pair_family_summary": record.get("pair_family_summary"),
            "vine_structure": record.get("vine_structure"),
        },
        "sampling": {
            "num_real_parent_rows": int(len(u_df)),
            "num_repeats": int(args.num_repeats),
            "num_generated_child_samples": int(n_generated),
            "seed": int(args.seed),
        },
        "overall_dependence": {
            "real_kendall_parent_1_child": real_kendall_p1_child,
            "generated_kendall_parent_1_child": gen_kendall_p1_child,
            "abs_error_kendall_parent_1_child": (
                None
                if real_kendall_p1_child is None or gen_kendall_p1_child is None
                else abs(real_kendall_p1_child - gen_kendall_p1_child)
            ),

            "real_kendall_parent_2_child": real_kendall_p2_child,
            "generated_kendall_parent_2_child": gen_kendall_p2_child,
            "abs_error_kendall_parent_2_child": (
                None
                if real_kendall_p2_child is None or gen_kendall_p2_child is None
                else abs(real_kendall_p2_child - gen_kendall_p2_child)
            ),

            "real_spearman_parent_1_child": real_spearman_p1_child,
            "generated_spearman_parent_1_child": gen_spearman_p1_child,
            "abs_error_spearman_parent_1_child": (
                None
                if real_spearman_p1_child is None or gen_spearman_p1_child is None
                else abs(real_spearman_p1_child - gen_spearman_p1_child)
            ),

            "real_spearman_parent_2_child": real_spearman_p2_child,
            "generated_spearman_parent_2_child": gen_spearman_p2_child,
            "abs_error_spearman_parent_2_child": (
                None
                if real_spearman_p2_child is None or gen_spearman_p2_child is None
                else abs(real_spearman_p2_child - gen_spearman_p2_child)
            ),
        },
        "conditional_bin_summary": {
            "mean_abs_bin_mean_error": mean_abs_bin_mean_error,
            "median_abs_bin_median_error": median_abs_bin_median_error,
        },
    }

    # Save outputs.
    mechanism_name = f"{child}__given__{parent1}__{parent2}"
    mechanism_name = mechanism_name.replace(" ", "_")

    summary_path = output_dir / f"{mechanism_name}_summary.json"
    bin_diag_path = output_dir / f"{mechanism_name}_bin_diagnostics.csv"
    real_path = output_dir / f"{mechanism_name}_real_validation_data.csv"
    generated_path = output_dir / f"{mechanism_name}_generated_validation_data.csv"

    save_json(summary, summary_path)
    bin_diag.to_csv(bin_diag_path, index=False)
    real_validation_df.to_csv(real_path, index=False)
    generated_validation_df.to_csv(generated_path, index=False)

    print("\nConditional-vine validation complete.")
    print(f"Summary saved to: {summary_path}")
    print(f"Bin diagnostics saved to: {bin_diag_path}")

    print("\nOverall dependence summary:")
    print(json.dumps(summary["overall_dependence"], indent=2))

    print("\nConditional bin diagnostics:")
    print(bin_diag.to_string(index=False))

    print("\nInterpretation guide:")
    print("- Generated child values should be inside (0, 1).")
    print("- Parent-child Kendall/Spearman signs should usually match the real data.")
    print("- Bin-wise generated child means/medians should broadly follow real conditional trends.")
    print("- This is a sanity check, not a formal goodness-of-fit proof.")


if __name__ == "__main__":
    main()