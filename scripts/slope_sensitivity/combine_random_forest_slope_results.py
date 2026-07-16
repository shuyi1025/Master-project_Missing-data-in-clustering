"""Combine the four split-slope random-forest sensitivity results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2] if SCRIPT_PATH.parents[1].name == "scripts" else SCRIPT_PATH.parents[1]
DEFAULT_RESULTS_DIR = ROOT / "outputs" / "slope_sensitivity"

METHODS = ["random_forest"]
SLOPES = {
    "neg050": -0.50,
    "neg075": -0.75,
    "neg200": -2.00,
    "neg250": -2.50,
}
SYNTHETIC_SIZES = [204, 2000, 5000]
OUTPUT_PREFIX = "sensitivity_random_forest_slope4"

DETAIL_KEY = [
    "base_random_state",
    "replication",
    "n_synthetic",
    "method",
    "mechanism",
    "target_missing_rate",
    "missingness_logit_slope",
]

METRIC_AGGREGATIONS = {
    "mean_observed_missing_rate": ("observed_missing_rate", "mean"),
    "sd_observed_missing_rate": ("observed_missing_rate", "std"),
    "min_observed_missing_rate": ("observed_missing_rate", "min"),
    "max_observed_missing_rate": ("observed_missing_rate", "max"),
    "mean_ari": ("ari_vs_full_data_kmeans", "mean"),
    "sd_ari": ("ari_vs_full_data_kmeans", "std"),
    "min_ari": ("ari_vs_full_data_kmeans", "min"),
    "max_ari": ("ari_vs_full_data_kmeans", "max"),
    "mean_centroid_error": ("centroid_error_mean", "mean"),
    "sd_centroid_error": ("centroid_error_mean", "std"),
    "min_centroid_error": ("centroid_error_mean", "min"),
    "max_centroid_error": ("centroid_error_mean", "max"),
    "mean_centroid_error_max": ("centroid_error_max", "mean"),
    "sd_centroid_error_max": ("centroid_error_max", "std"),
    "min_centroid_error_max": ("centroid_error_max", "min"),
    "max_centroid_error_max": ("centroid_error_max", "max"),
    "mean_cluster_size_error": ("cluster_size_error_mean_abs", "mean"),
    "sd_cluster_size_error": ("cluster_size_error_mean_abs", "std"),
    "min_cluster_size_error": ("cluster_size_error_mean_abs", "min"),
    "max_cluster_size_error": ("cluster_size_error_mean_abs", "max"),
    "mean_silhouette": ("silhouette", "mean"),
    "sd_silhouette": ("silhouette", "std"),
    "min_silhouette": ("silhouette", "min"),
    "max_silhouette": ("silhouette", "max"),
    "mean_runtime_seconds": ("runtime_seconds", "mean"),
    "sd_runtime_seconds": ("runtime_seconds", "std"),
    "min_runtime_seconds": ("runtime_seconds", "min"),
    "max_runtime_seconds": ("runtime_seconds", "max"),
}


def expected_result_path(results_dir: Path, method: str, slope_label: str, size: int) -> Path:
    return results_dir / f"sensitivity_{method}_slope_{slope_label}_results_n{size}.csv"


def validate_frame(
    frame: pd.DataFrame,
    path: Path,
    expected_method: str,
    expected_slope: float,
    expected_size: int,
) -> None:
    missing_columns = [column for column in DETAIL_KEY if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError(f"{path.name} contains no result rows.")

    methods = set(frame["method"].dropna().astype(str))
    if methods != {expected_method}:
        raise ValueError(f"{path.name} has methods {sorted(methods)}, expected {expected_method}.")

    sizes = set(pd.to_numeric(frame["n_synthetic"], errors="raise").astype(int))
    if sizes != {expected_size}:
        raise ValueError(f"{path.name} has sample sizes {sorted(sizes)}, expected {expected_size}.")

    slopes = pd.to_numeric(frame["missingness_logit_slope"], errors="raise").to_numpy()
    if not np.all(np.isclose(slopes, expected_slope)):
        observed = sorted(set(slopes))
        raise ValueError(f"{path.name} has slopes {observed}, expected {expected_slope}.")


def load_split_results(results_dir: Path, allow_incomplete: bool) -> pd.DataFrame:
    frames = []
    missing_paths = []

    for size in SYNTHETIC_SIZES:
        for method in METHODS:
            for slope_label, slope in SLOPES.items():
                path = expected_result_path(results_dir, method, slope_label, size)
                if not path.exists():
                    missing_paths.append(path)
                    continue
                frame = pd.read_csv(path)
                validate_frame(frame, path, method, slope, size)
                frame["split_source_file"] = path.name
                frames.append(frame)

    if missing_paths and not allow_incomplete:
        missing_list = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Some split-slope result files are missing. Wait for all HPC jobs to finish, "
            f"or use --allow-incomplete.\n{missing_list}"
        )
    if not frames:
        raise FileNotFoundError(f"No split random-forest result CSVs found in {results_dir}")

    results = pd.concat(frames, ignore_index=True)
    duplicated = results.duplicated(subset=DETAIL_KEY, keep=False)
    if duplicated.any():
        examples = results.loc[duplicated, DETAIL_KEY].head(10).to_dict("records")
        raise ValueError(f"Duplicate detailed result rows detected: {examples}")
    return results.sort_values(DETAIL_KEY).reset_index(drop=True)


def make_summaries(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        results.groupby(
            ["method", "mechanism", "target_missing_rate", "missingness_logit_slope"]
        )
        .agg(n_runs=("ari_vs_full_data_kmeans", "size"), **METRIC_AGGREGATIONS)
        .reset_index()
    )
    seed_summary = (
        results.groupby(
            [
                "base_random_state",
                "method",
                "mechanism",
                "target_missing_rate",
                "missingness_logit_slope",
            ]
        )
        .agg(n_replications=("ari_vs_full_data_kmeans", "size"), **METRIC_AGGREGATIONS)
        .reset_index()
    )
    return summary, seed_summary


def save_outputs(results: pd.DataFrame, results_dir: Path, output_prefix: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    all_details = []
    all_summaries = []
    all_seed_summaries = []

    for size, size_results in results.groupby("n_synthetic", sort=True):
        size = int(size)
        summary, seed_summary = make_summaries(size_results)

        detail_path = results_dir / f"{output_prefix}_results_n{size}.csv"
        summary_path = results_dir / f"{output_prefix}_summary_n{size}.csv"
        seed_path = results_dir / f"{output_prefix}_summary_by_seed_n{size}.csv"
        size_results.to_csv(detail_path, index=False)
        summary.to_csv(summary_path, index=False)
        seed_summary.to_csv(seed_path, index=False)

        all_details.append(size_results)
        all_summaries.append(summary.assign(dataset_n=size))
        all_seed_summaries.append(seed_summary.assign(dataset_n=size))
        print(f"n={size}: {len(size_results)} detailed rows -> {detail_path.name}")

    pd.concat(all_details, ignore_index=True).to_csv(
        results_dir / f"{output_prefix}_results_all_sizes.csv", index=False
    )
    pd.concat(all_summaries, ignore_index=True).to_csv(
        results_dir / f"{output_prefix}_summary_all_sizes.csv", index=False
    )
    pd.concat(all_seed_summaries, ignore_index=True).to_csv(
        results_dir / f"{output_prefix}_summary_by_seed_all_sizes.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine the four split-slope random-forest sensitivity results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Input/output directory. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Combine available files even if one or more expected jobs are unfinished.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    results = load_split_results(results_dir, allow_incomplete=args.allow_incomplete)
    save_outputs(results, results_dir, args.output_prefix)
    print(f"Combined {len(results)} rows from {results['split_source_file'].nunique()} files.")


if __name__ == "__main__":
    main()
