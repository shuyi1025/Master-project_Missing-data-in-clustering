"""Run imputation sensitivity checks across random seeds and slopes."""

import argparse
import os
import time
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "sensitivity_median_seed_slope"
SYNTHETIC_SOURCES = {
    "204": ROOT / "outputs" / "simulation_gmm_kmeans204" / "synthetic_complete_standardised.csv",
    "2000": ROOT / "outputs" / "simulation_gmm_kmeans2000" / "synthetic_complete_standardised.csv",
    "5000": ROOT / "outputs" / "simulation_gmm_kmeans5000" / "synthetic_complete_standardised.csv",
}

FEATURE_COLS = [
    "totalmins",
    "logindays",
    "loginwks",
    "pageviews",
    "posts",
    "ptp",
    "ate",
    "totaldays",
    "act_wkpv",
    "dur_wkpv",
    "act_ptp",
    "dur_ptp",
    "act_ate",
    "dur_ate",
    "sdwkpv",
    "sdptp",
    "sdate",
    "rate_wkpv",
    "rate_ptp",
    "rate_ate",
]

TRUE_LABEL_COL = "true_gmm_component"
BENCHMARK_LABEL_COL = "kmeans_full_data_cluster"
K = 5
KMEANS_N_INIT = 50
KPOD_MAX_ITER = 50
KPOD_TOL = 1e-4
MAR_DRIVER_COLS = ["totalmins", "pageviews", "posts"]
DEFAULT_RATES = [0.30, 0.40, 0.50]
DEFAULT_SEEDS = [101, 123, 202, 404, 808]
DEFAULT_SLOPES = [-0.50, -0.75, -1.25, -2.00, -2.50]
DEFAULT_MECHANISMS = ["MAR", "MNAR"]
DEFAULT_METHODS = ["median", "kpod"]


def scenario_seed(base_random_state, *parts):
    value = int(base_random_state)
    for part in parts:
        if isinstance(part, str):
            part = sum(ord(char) for char in part)
        value = (value * 1_000_003 + int(part)) % (2**32 - 1)
    return value


def zero_based_labels(labels):
    labels = pd.Series(labels).astype(int).to_numpy()
    if labels.min() == 1 and labels.max() == K:
        return labels - 1
    return labels


def load_synthetic_complete_data(synthetic_size):
    source_path = SYNTHETIC_SOURCES[str(synthetic_size)]
    if not source_path.exists():
        raise FileNotFoundError(
            f"Synthetic source not found: {source_path}. Generate it before running sensitivity checks."
        )

    df = pd.read_csv(source_path)
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected synthetic feature columns: {missing_cols}")

    x_complete = df[FEATURE_COLS].astype(float)
    true_labels = zero_based_labels(df[TRUE_LABEL_COL])
    benchmark_labels = zero_based_labels(df[BENCHMARK_LABEL_COL])
    return source_path, x_complete, true_labels, benchmark_labels


def run_kmeans(x, random_state):
    model = KMeans(n_clusters=K, random_state=random_state, n_init=KMEANS_N_INIT)
    labels = model.fit_predict(x)
    return model, labels


def make_reference_kmeans_from_labels(x_complete, labels):
    centers = []
    for k in range(K):
        rows = x_complete.loc[labels == k]
        if rows.empty:
            raise ValueError(f"Benchmark cluster {k + 1} has no rows.")
        centers.append(rows.mean(axis=0).to_numpy())
    return SimpleNamespace(cluster_centers_=np.vstack(centers)), labels


def standardize_score(score):
    score = np.asarray(score, dtype=float)
    sd = score.std()
    if sd == 0 or np.isnan(sd):
        return np.zeros_like(score)
    return (score - score.mean()) / sd


def calibrate_intercept(score, target_rate, slope):
    low, high = -20.0, 20.0
    for _ in range(80):
        mid = (low + high) / 2
        probs = 1 / (1 + np.exp(-(mid + slope * score)))
        if probs.mean() < target_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def logistic_missingness_prob(driver, target_rate, slope):
    score = standardize_score(driver)
    intercept = calibrate_intercept(score, target_rate, slope)
    return 1 / (1 + np.exp(-(intercept + slope * score)))


def inject_missingness(x_complete, mechanism, rate, random_state, slope):
    rng = np.random.default_rng(random_state)
    mechanism = mechanism.upper()

    if mechanism == "MCAR":
        mask = rng.random(x_complete.shape) < rate
    elif mechanism == "MAR":
        mask = np.zeros(x_complete.shape, dtype=bool)
        for j, target_col in enumerate(x_complete.columns):
            driver_cols = [
                col for col in MAR_DRIVER_COLS
                if col in x_complete.columns and col != target_col
            ]
            if not driver_cols:
                driver_cols = [col for col in x_complete.columns if col != target_col]
            probs = logistic_missingness_prob(x_complete[driver_cols].mean(axis=1), rate, slope)
            mask[:, j] = rng.random(len(x_complete)) < probs
    elif mechanism == "MNAR":
        mask = np.zeros(x_complete.shape, dtype=bool)
        for j, target_col in enumerate(x_complete.columns):
            probs = logistic_missingness_prob(x_complete[target_col], rate, slope)
            mask[:, j] = rng.random(len(x_complete)) < probs
    else:
        raise ValueError(f"Unknown missingness mechanism: {mechanism}")

    mask = pd.DataFrame(mask, columns=x_complete.columns, index=x_complete.index)
    return x_complete.mask(mask), mask


def as_frame(values, template):
    return pd.DataFrame(values, columns=template.columns, index=template.index)


def impute_median(x_missing):
    start = time.perf_counter()
    imputer = SimpleImputer(strategy="median")
    completed = as_frame(imputer.fit_transform(x_missing), x_missing)
    return {
        "completed": completed,
        "model": None,
        "labels": None,
        "runtime_seconds": time.perf_counter() - start,
    }


def initial_median_fill(x_missing):
    imputer = SimpleImputer(strategy="median")
    return as_frame(imputer.fit_transform(x_missing), x_missing)


def impute_kpod(x_missing, random_state):
    start = time.perf_counter()
    x_filled = initial_median_fill(x_missing)
    missing_mask = x_missing.isna().to_numpy()
    previous_centers = None

    for _ in range(KPOD_MAX_ITER):
        model = KMeans(n_clusters=K, random_state=random_state, n_init=KMEANS_N_INIT)
        labels = model.fit_predict(x_filled)
        centers = model.cluster_centers_

        values = x_filled.to_numpy(copy=True)
        rows, cols = np.where(missing_mask)
        values[rows, cols] = centers[labels[rows], cols]
        x_filled = as_frame(values, x_missing)

        if previous_centers is not None and np.linalg.norm(centers - previous_centers) < KPOD_TOL:
            break
        previous_centers = centers.copy()

    final_model, final_labels = run_kmeans(x_filled, random_state=random_state)
    return {
        "completed": x_filled,
        "model": final_model,
        "labels": final_labels,
        "runtime_seconds": time.perf_counter() - start,
    }


def impute_with_method(x_missing, method, random_state):
    if method == "median":
        return impute_median(x_missing)
    if method == "kpod":
        return impute_kpod(x_missing, random_state=random_state)
    raise ValueError(f"Unknown imputation method: {method}")


def cluster_proportions(labels):
    counts = np.bincount(labels, minlength=K)
    return counts / counts.sum()


def best_label_permutation(reference_centers, estimated_centers):
    best_perm = None
    best_cost = np.inf
    for perm in permutations(range(K)):
        cost = np.linalg.norm(reference_centers - estimated_centers[list(perm)], axis=1).sum()
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    return list(best_perm)


def compute_metrics(x_completed, estimated_model, estimated_labels, benchmark_model, benchmark_labels):
    perm = best_label_permutation(
        benchmark_model.cluster_centers_,
        estimated_model.cluster_centers_,
    )
    matched_centers = estimated_model.cluster_centers_[perm]
    centroid_distances = np.linalg.norm(
        benchmark_model.cluster_centers_ - matched_centers,
        axis=1,
    )
    estimated_props = cluster_proportions(estimated_labels)[perm]
    benchmark_props = cluster_proportions(benchmark_labels)

    return {
        "ari_vs_full_data_kmeans": adjusted_rand_score(benchmark_labels, estimated_labels),
        "centroid_error_mean": centroid_distances.mean(),
        "centroid_error_max": centroid_distances.max(),
        "cluster_size_error_mean_abs": np.abs(benchmark_props - estimated_props).mean(),
        "silhouette": silhouette_score(x_completed, estimated_labels)
        if len(np.unique(estimated_labels)) > 1
        else np.nan,
    }


def save_summaries(results, output_stem, synthetic_size):
    metric_aggregations = {
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

    summary = (
        results.groupby(["method", "mechanism", "target_missing_rate", "missingness_logit_slope"])
        .agg(
            n_runs=("ari_vs_full_data_kmeans", "size"),
            **metric_aggregations,
        )
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
        .agg(
            n_replications=("ari_vs_full_data_kmeans", "size"),
            **metric_aggregations,
        )
        .reset_index()
    )

    summary_path = OUTPUT_DIR / f"{output_stem}_summary_n{synthetic_size}.csv"
    seed_summary_path = OUTPUT_DIR / f"{output_stem}_summary_by_seed_n{synthetic_size}.csv"
    summary.to_csv(summary_path, index=False)
    seed_summary.to_csv(seed_summary_path, index=False)
    return summary_path, seed_summary_path


def run_one_synthetic_size(args, synthetic_size):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_path, x_complete, true_gmm_labels, benchmark_labels = load_synthetic_complete_data(
        synthetic_size
    )
    benchmark_model, benchmark_labels = make_reference_kmeans_from_labels(
        x_complete,
        benchmark_labels,
    )
    reference_ari = adjusted_rand_score(true_gmm_labels, benchmark_labels)

    output_stem = args.output_prefix or "sensitivity"
    result_path = OUTPUT_DIR / f"{output_stem}_results_n{synthetic_size}.csv"

    print("=" * 72)
    print(f"Synthetic size: {synthetic_size}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Synthetic source: {source_path}")
    print(f"Loaded complete synthetic records: {len(x_complete)}")
    print(f"Writing detailed results to: {result_path}")
    print("Note: slope controls the strength of MAR/MNAR logistic missingness.")

    rows = []
    for base_seed in args.base_seeds:
        for slope in args.slopes:
            for b in range(1, args.b + 1):
                for mechanism in args.mechanisms:
                    for rate in args.rates:
                        missingness_seed = scenario_seed(
                            base_seed,
                            "missingness",
                            b,
                            int(rate * 1000),
                            mechanism,
                        )
                        x_missing, mask = inject_missingness(
                            x_complete,
                            mechanism=mechanism,
                            rate=rate,
                            random_state=missingness_seed,
                            slope=slope,
                        )
                        for method in args.methods:
                            method_seed = scenario_seed(
                                base_seed,
                                "method",
                                b,
                                int(rate * 1000),
                                mechanism,
                                method,
                            )
                            method_result = impute_with_method(
                                x_missing,
                                method=method,
                                random_state=method_seed,
                            )
                            x_completed = method_result["completed"]
                            model = method_result["model"]
                            labels = method_result["labels"]

                            if model is None or labels is None:
                                model, labels = run_kmeans(
                                    x_completed,
                                    random_state=scenario_seed(
                                        base_seed,
                                        "kmeans",
                                        b,
                                        int(rate * 1000),
                                        mechanism,
                                        method,
                                    ),
                                )

                            metrics = compute_metrics(
                                x_completed,
                                model,
                                labels,
                                benchmark_model,
                                benchmark_labels,
                            )
                            rows.append(
                                {
                                    "base_random_state": base_seed,
                                    "replication": b,
                                    "n_synthetic": len(x_complete),
                                    "synthetic_source": str(source_path),
                                    "mechanism": mechanism,
                                    "target_missing_rate": rate,
                                    "observed_missing_rate": float(mask.to_numpy().mean()),
                                    "missingness_logit_slope": slope,
                                    "method": method,
                                    "runtime_seconds": method_result["runtime_seconds"],
                                    "ari_true_gmm_vs_benchmark": reference_ari,
                                    **metrics,
                                }
                            )

                print(f"seed={base_seed} slope={slope} replication={b} complete")

    results = pd.DataFrame(rows)
    results.to_csv(result_path, index=False)
    summary_path, seed_summary_path = save_summaries(results, output_stem, synthetic_size)
    print(f"Saved summary to: {summary_path}")
    print(f"Saved seed-level summary to: {seed_summary_path}")
    return results


def run(args):
    if args.synthetic_size == "all":
        synthetic_sizes = sorted(SYNTHETIC_SOURCES, key=int)
    else:
        synthetic_sizes = [args.synthetic_size]

    all_results = []
    for synthetic_size in synthetic_sizes:
        all_results.append(run_one_synthetic_size(args, synthetic_size))

    return pd.concat(all_results, ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sensitivity check for imputation methods across seeds and slopes."
    )
    parser.add_argument(
        "--synthetic-size",
        choices=sorted(SYNTHETIC_SOURCES) + ["all"],
        default="5000",
        help="Synthetic dataset to use, or 'all' to run 204, 2000, and 5000. Default: 5000.",
    )
    parser.add_argument("--methods", nargs="+", choices=["median", "kpod"], default=DEFAULT_METHODS)
    parser.add_argument("--mechanisms", nargs="+", choices=["MAR", "MNAR"], default=DEFAULT_MECHANISMS)
    parser.add_argument("--rates", nargs="+", type=float, default=DEFAULT_RATES)
    parser.add_argument("--base-seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--slopes", nargs="+", type=float, default=DEFAULT_SLOPES)
    parser.add_argument("--b", type=int, default=30, help="Replications per seed/slope/rate.")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Short output stem. Default creates sensitivity_*_n<size>.csv files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    start = time.perf_counter()
    out = run(parse_args())
    print("\nSensitivity run complete.")
    print(
        out.groupby(["method", "mechanism", "target_missing_rate", "missingness_logit_slope"])[
            [
                "ari_vs_full_data_kmeans",
                "silhouette",
                "centroid_error_mean",
                "centroid_error_max",
                "cluster_size_error_mean_abs",
                "observed_missing_rate",
                "runtime_seconds",
            ]
        ]
        .mean()
        .round(4)
    )
    print(f"Elapsed seconds: {time.perf_counter() - start:.1f}")
