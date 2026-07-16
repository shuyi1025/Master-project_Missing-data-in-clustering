import argparse
import os
import time
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace


def detect_num_workers():
    try:
        pbs_workers = int(os.getenv("PBS_NP", "0"))
    except ValueError:
        pbs_workers = 0
    return pbs_workers if pbs_workers > 0 else max(1, (os.cpu_count() or 1) // 4)


NUM_WORKERS = detect_num_workers()
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(NUM_WORKERS))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import adjusted_rand_score, silhouette_samples


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "simulation2000"
SYNTHETIC_SOURCE_PATH = (
    ROOT / "outputs" / "simulation_gmm_kmeans2000" / "synthetic_complete_standardised.csv"
)

FEATURE_COLS = [
    "totalmins", "logindays", "loginwks", "pageviews", "posts",
    "ptp", "ate", "totaldays", "act_wkpv", "dur_wkpv",
    "act_ptp", "dur_ptp", "act_ate", "dur_ate", "sdwkpv",
    "sdptp", "sdate", "rate_wkpv", "rate_ptp", "rate_ate",
]

METHOD = "mice_pmm"
TRUE_LABEL_COL = "true_gmm_component"
BENCHMARK_LABEL_COL = "kmeans_full_data_cluster"
K = 5
B = 500
MISSINGNESS_MECHANISM = "MCAR"
MISSINGNESS_RATES = [0.10, 0.20, 0.30, 0.40, 0.50]
RANDOM_STATE = 123
KMEANS_N_INIT = 50
MAR_DRIVER_COLS = ["totalmins", "pageviews", "posts"]
MISSINGNESS_LOGIT_SLOPE = -1.25
MICE_N_IMPUTATIONS = 10
MICE_MAX_ITER = 5
PMM_DONORS = 5
SCALAR_METRICS = [
    "ari_vs_full_data_kmeans",
    "centroid_error_mean",
    "centroid_error_max",
    "cluster_size_error_mean_abs",
    "silhouette",
]


def seed(*parts):
    value = RANDOM_STATE
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


def load_synthetic_complete_data():
    if not SYNTHETIC_SOURCE_PATH.exists():
        raise FileNotFoundError(
            "Synthetic complete data file not found. Run "
            "scripts/simulation_2000/simulation_gmm_kmeans.py first. Expected: "
            f"{SYNTHETIC_SOURCE_PATH}"
        )
    df = pd.read_csv(SYNTHETIC_SOURCE_PATH)
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected synthetic feature columns: {missing_cols}")
    return (
        df[FEATURE_COLS].astype(float),
        zero_based_labels(df[TRUE_LABEL_COL]),
        zero_based_labels(df[BENCHMARK_LABEL_COL]),
    )


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


def calibrate_intercept(score, target_rate):
    low, high = -20.0, 20.0
    for _ in range(80):
        mid = (low + high) / 2
        probs = 1 / (1 + np.exp(-(mid + MISSINGNESS_LOGIT_SLOPE * score)))
        if probs.mean() < target_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def logistic_missingness_prob(driver, target_rate):
    score = standardize_score(driver)
    intercept = calibrate_intercept(score, target_rate)
    return 1 / (1 + np.exp(-(intercept + MISSINGNESS_LOGIT_SLOPE * score)))


def inject_missingness(x_complete, mechanism, rate, random_state):
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
            probs = logistic_missingness_prob(x_complete[driver_cols].mean(axis=1), rate)
            mask[:, j] = rng.random(len(x_complete)) < probs
    elif mechanism == "MNAR":
        mask = np.zeros(x_complete.shape, dtype=bool)
        for j, target_col in enumerate(x_complete.columns):
            probs = logistic_missingness_prob(x_complete[target_col], rate)
            mask[:, j] = rng.random(len(x_complete)) < probs
    else:
        raise ValueError(f"Unknown missingness mechanism: {mechanism}")
    mask = pd.DataFrame(mask, columns=x_complete.columns, index=x_complete.index)
    return x_complete.mask(mask), mask


def as_frame(values, template):
    return pd.DataFrame(values, columns=template.columns, index=template.index)


def initial_median_fill(x_missing):
    imputer = SimpleImputer(strategy="median")
    return as_frame(imputer.fit_transform(x_missing), x_missing)


def pmm_once(x_missing, random_state):
    rng = np.random.default_rng(random_state)
    x_filled = initial_median_fill(x_missing)
    missing_mask = x_missing.isna()

    for _ in range(MICE_MAX_ITER):
        for target_col in x_missing.columns:
            missing_rows = missing_mask[target_col].to_numpy()
            if not missing_rows.any():
                continue

            observed_rows = ~missing_rows
            predictor_cols = [col for col in x_missing.columns if col != target_col]
            x_obs = x_filled.loc[observed_rows, predictor_cols]
            y_obs = x_filled.loc[observed_rows, target_col]
            x_mis = x_filled.loc[missing_rows, predictor_cols]

            model = BayesianRidge()
            model.fit(x_obs, y_obs)
            pred_obs = model.predict(x_obs)
            pred_mis = model.predict(x_mis)
            donor_values = y_obs.to_numpy()

            imputed = []
            for pred in pred_mis:
                donor_idx = np.argsort(np.abs(pred_obs - pred))[:PMM_DONORS]
                imputed.append(rng.choice(donor_values[donor_idx]))
            x_filled.loc[missing_rows, target_col] = imputed

    return x_filled


def impute_mice_pmm(x_missing, random_state, n_imputations=MICE_N_IMPUTATIONS):
    start = time.perf_counter()
    completed = [
        pmm_once(x_missing, random_state + m * 1009)
        for m in range(n_imputations)
    ]
    return {
        "completed_datasets": completed,
        "runtime_seconds": time.perf_counter() - start,
    }


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
    if len(np.unique(estimated_labels)) > 1:
        sample_silhouettes = silhouette_samples(x_completed, estimated_labels)
        silhouette = float(sample_silhouettes.mean())
        silhouette_within_variance = float(sample_silhouettes.var(ddof=1) / len(sample_silhouettes))
    else:
        silhouette = np.nan
        silhouette_within_variance = np.nan
    n = len(estimated_labels)
    cluster_size_error_within_variance = float(
        np.sum(estimated_props * (1 - estimated_props) / n) / (K**2)
    )
    return {
        "ari_vs_full_data_kmeans": adjusted_rand_score(benchmark_labels, estimated_labels),
        "ari_vs_full_data_kmeans_within_variance": np.nan,
        "centroid_error_mean": centroid_distances.mean(),
        "centroid_error_mean_within_variance": np.nan,
        "centroid_error_max": centroid_distances.max(),
        "centroid_error_max_within_variance": np.nan,
        "cluster_size_error_mean_abs": np.abs(benchmark_props - estimated_props).mean(),
        "cluster_size_error_mean_abs_within_variance": cluster_size_error_within_variance,
        "silhouette": silhouette,
        "silhouette_within_variance": silhouette_within_variance,
    }


def rubins_pool(estimates, within_variances=None):
    estimates = np.asarray(estimates, dtype=float)
    valid = np.isfinite(estimates)
    estimates = estimates[valid]
    if within_variances is None:
        within_variances = np.zeros_like(estimates)
        within_variance_available = False
    else:
        within_variances = np.asarray(within_variances, dtype=float)[valid]
        within_variance_available = bool(np.isfinite(within_variances).any())
        within_variances = np.nan_to_num(within_variances, nan=0.0)

    m = len(estimates)
    if m == 0:
        return {
            "n_imputations": 0,
            "estimate": np.nan,
            "within_variance": np.nan,
            "between_variance": np.nan,
            "total_variance": np.nan,
            "total_se": np.nan,
            "within_variance_available": within_variance_available,
        }

    estimate = float(estimates.mean())
    within_variance = float(within_variances.mean()) if m else np.nan
    between_variance = float(estimates.var(ddof=1)) if m > 1 else 0.0
    total_variance = within_variance + (1 + 1 / m) * between_variance
    return {
        "n_imputations": m,
        "estimate": estimate,
        "within_variance": within_variance,
        "between_variance": between_variance,
        "total_variance": total_variance,
        "total_se": float(np.sqrt(total_variance)),
        "within_variance_available": within_variance_available,
    }


def rubins_pool_wide(metric_rows):
    metric_df = pd.DataFrame(metric_rows)
    pooled = {}
    for metric in SCALAR_METRICS:
        within_col = f"{metric}_within_variance"
        result = rubins_pool(
            metric_df[metric],
            metric_df[within_col] if within_col in metric_df.columns else None,
        )
        pooled[metric] = result["estimate"]
        pooled[f"{metric}_within_variance"] = result["within_variance"]
        pooled[f"{metric}_between_variance"] = result["between_variance"]
        pooled[f"{metric}_total_variance"] = result["total_variance"]
        pooled[f"{metric}_rubin_se"] = result["total_se"]
        pooled[f"{metric}_within_variance_available"] = result["within_variance_available"]
    return pooled


def build_poolable_estimates(x_completed, estimated_model, estimated_labels, benchmark_model, benchmark_labels, metrics):
    perm = best_label_permutation(
        benchmark_model.cluster_centers_,
        estimated_model.cluster_centers_,
    )
    matched_centers = estimated_model.cluster_centers_[perm]
    estimated_counts = np.bincount(estimated_labels, minlength=K)[perm]
    estimated_props = estimated_counts / estimated_counts.sum()
    rows = []

    for metric in SCALAR_METRICS:
        rows.append(
            {
                "metric": metric,
                "component": "overall",
                "feature": "",
                "estimate": metrics[metric],
                "within_variance": metrics.get(f"{metric}_within_variance", np.nan),
            }
        )

    n = len(estimated_labels)
    for cluster_idx, (count, prop) in enumerate(zip(estimated_counts, estimated_props), start=1):
        rows.append(
            {
                "metric": "cluster_size",
                "component": f"cluster_{cluster_idx}",
                "feature": "",
                "estimate": float(count),
                "within_variance": float(n * prop * (1 - prop)),
            }
        )
        rows.append(
            {
                "metric": "cluster_proportion",
                "component": f"cluster_{cluster_idx}",
                "feature": "",
                "estimate": float(prop),
                "within_variance": float(prop * (1 - prop) / n),
            }
        )

    for benchmark_cluster_idx, estimated_cluster_idx in enumerate(perm):
        members = x_completed.loc[estimated_labels == estimated_cluster_idx]
        if len(members) > 1:
            centroid_variances = members.var(axis=0, ddof=1) / len(members)
        else:
            centroid_variances = pd.Series(0.0, index=x_completed.columns)
        for feature_idx, feature in enumerate(x_completed.columns):
            rows.append(
                {
                    "metric": "centroid",
                    "component": f"cluster_{benchmark_cluster_idx + 1}",
                    "feature": feature,
                    "estimate": float(matched_centers[benchmark_cluster_idx, feature_idx]),
                    "within_variance": float(centroid_variances[feature]),
                }
            )
    return rows


def pool_long_results(poolable_results):
    poolable_df = pd.DataFrame(poolable_results)
    if poolable_df.empty:
        return poolable_df

    group_cols = [
        "replication",
        "mechanism",
        "target_missing_rate",
        "method",
        "metric",
        "component",
        "feature",
    ]
    rows = []
    for keys, group in poolable_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row.update(rubins_pool(group["estimate"], group["within_variance"]))
        rows.append(row)
    return pd.DataFrame(rows)


def run_simulation_task(
    b, mechanism, rate, n_imputations, x_complete, benchmark_model, benchmark_labels, reference_ari
):
    x_missing, mask = inject_missingness(
        x_complete, mechanism, rate, seed("missingness", b, int(rate * 1000), mechanism)
    )
    method_result = impute_mice_pmm(
        x_missing,
        seed("method", b, int(rate * 1000), mechanism, METHOD),
        n_imputations=n_imputations,
    )
    metric_rows = []
    imputation_rows = []
    poolable_rows = []
    for m, x_completed in enumerate(method_result["completed_datasets"], start=1):
        model, labels = run_kmeans(
            x_completed, seed("kmeans", b, int(rate * 1000), mechanism, METHOD, m)
        )
        metrics = compute_metrics(x_completed, model, labels, benchmark_model, benchmark_labels)
        metric_rows.append(metrics)
        context = {
            "replication": b,
            "n_synthetic": len(x_complete),
            "synthetic_source": str(SYNTHETIC_SOURCE_PATH),
            "mechanism": mechanism,
            "target_missing_rate": rate,
            "observed_missing_rate": float(mask.to_numpy().mean()),
            "method": METHOD,
            "imputation": m,
        }
        imputation_rows.append({**context, **metrics})
        for poolable in build_poolable_estimates(
            x_completed, model, labels, benchmark_model, benchmark_labels, metrics
        ):
            poolable_rows.append({**context, **poolable})
    pooled = rubins_pool_wide(metric_rows)
    result = {
        "replication": b,
        "n_synthetic": len(x_complete),
        "synthetic_source": str(SYNTHETIC_SOURCE_PATH),
        "mechanism": mechanism,
        "target_missing_rate": rate,
        "observed_missing_rate": float(mask.to_numpy().mean()),
        "method": METHOD,
        "n_imputed_datasets": len(method_result["completed_datasets"]),
        "runtime_seconds": method_result["runtime_seconds"],
        "ari_true_gmm_vs_benchmark": reference_ari,
        **pooled,
    }
    return result, imputation_rows, poolable_rows


def summarise_results(results):
    summary = (
        results.groupby(["mechanism", "target_missing_rate", "method"])
        .agg(
            mean_ari=("ari_vs_full_data_kmeans", "mean"),
            sd_ari=("ari_vs_full_data_kmeans", "std"),
            mean_ari_rubin_se=("ari_vs_full_data_kmeans_rubin_se", "mean"),
            mean_centroid_error=("centroid_error_mean", "mean"),
            mean_centroid_error_rubin_se=("centroid_error_mean_rubin_se", "mean"),
            mean_cluster_size_error=("cluster_size_error_mean_abs", "mean"),
            mean_cluster_size_error_rubin_se=("cluster_size_error_mean_abs_rubin_se", "mean"),
            mean_silhouette=("silhouette", "mean"),
            mean_silhouette_rubin_se=("silhouette_rubin_se", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .reset_index()
    )
    summary_path = OUTPUT_DIR / f"simulation_summary_{METHOD}_mcar.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def run(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mechanisms = [MISSINGNESS_MECHANISM]
    rates = args.rates or MISSINGNESS_RATES
    b_repetitions = args.b or B
    n_imputations = args.m or MICE_N_IMPUTATIONS
    output_name = args.output or f"simulation_results_{METHOD}_mcar.csv"
    if args.smoke:
        rates = args.rates or [0.10]
        b_repetitions = args.b or 1
        output_name = args.output or f"simulation_results_{METHOD}_mcar_smoke.csv"

    x_complete, true_gmm_labels, benchmark_labels = load_synthetic_complete_data()
    benchmark_model, benchmark_labels = make_reference_kmeans_from_labels(x_complete, benchmark_labels)
    reference_ari = adjusted_rand_score(true_gmm_labels, benchmark_labels)
    result_path = OUTPUT_DIR / output_name
    print(f"Method: {METHOD}")
    print(f"Loaded complete synthetic records: {len(x_complete)}")
    print(f"Synthetic source: {SYNTHETIC_SOURCE_PATH}")
    print(f"Writing results to: {result_path}")
    print(f"PMM imputations per missing dataset: M={n_imputations}")

    tasks = [
        (b, mechanism, rate)
        for b in range(1, b_repetitions + 1)
        for mechanism in mechanisms
        for rate in rates
    ]
    print(f"Using {NUM_WORKERS} workers for parallel imputation")
    task_results = Parallel(n_jobs=NUM_WORKERS, verbose=10)(
        delayed(run_simulation_task)(
            b, mechanism, rate, n_imputations, x_complete, benchmark_model, benchmark_labels, reference_ari
        )
        for b, mechanism, rate in tasks
    )
    rows = [result for result, _, _ in task_results]
    imputation_metric_rows = [row for _, task_rows, _ in task_results for row in task_rows]
    poolable_rows = [row for _, _, task_rows in task_results for row in task_rows]
    results = pd.DataFrame(rows)
    results.to_csv(result_path, index=False)
    result_stem = Path(output_name).stem
    imputation_metrics = pd.DataFrame(imputation_metric_rows)
    imputation_metric_path = OUTPUT_DIR / f"{result_stem}_imputation_metrics.csv"
    imputation_metrics.to_csv(imputation_metric_path, index=False)
    pooled_long = pool_long_results(poolable_rows)
    pooled_long_path = OUTPUT_DIR / f"{result_stem}_rubins_pooled.csv"
    pooled_long.to_csv(pooled_long_path, index=False)
    print(f"Saved per-imputation metrics to: {imputation_metric_path}")
    print(f"Saved Rubin pooled estimates to: {pooled_long_path}")
    print(f"Saved summary to: {summarise_results(results)}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run the complete MICE PMM simulation script for MCAR.")
    parser.add_argument("--smoke", action="store_true", help="Run B=1 at 10%% missingness.")
    parser.add_argument("--b", type=int, default=None, help=f"Replications. Default: {B}.")
    parser.add_argument("--m", type=int, default=None, help=f"PMM imputations. Default: {MICE_N_IMPUTATIONS}.")
    parser.add_argument("--rates", nargs="+", type=float, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    start = time.perf_counter()
    out = run(parse_args())
    print("\nSimulation complete.")
    print(
        out.groupby(["mechanism", "target_missing_rate", "method"])[
            "ari_vs_full_data_kmeans"
        ]
        .mean()
        .round(3)
    )
    print(f"Elapsed seconds: {time.perf_counter() - start:.1f}")
