import argparse
import time
import warnings
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

import run_random_forest_mcar as simulation


LOCAL_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA = (
    LOCAL_ROOT / "outputs" / "simulation_gmm_kmeans" / "synthetic_complete_standardised.csv"
)
CONFIGS = [(5, 1e-3), (10, 1e-2), (20, 1e-2), (20, 5e-2)]


def test_config(x_complete, mechanism, rate, max_iter, tol, min_samples_leaf):
    x_missing, _ = simulation.inject_missingness(
        x_complete,
        mechanism,
        rate,
        simulation.seed("missingness", 1, int(rate * 1000), mechanism),
    )
    estimator = RandomForestRegressor(
        n_estimators=simulation.RF_N_ESTIMATORS,
        random_state=simulation.seed("diagnostic", mechanism, int(rate * 1000), max_iter),
        n_jobs=1,
        min_samples_leaf=min_samples_leaf,
    )
    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        tol=tol,
        random_state=simulation.seed("diagnostic-imputer", mechanism, int(rate * 1000)),
        initial_strategy="median",
        skip_complete=True,
    )
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        imputer.fit_transform(x_missing)
    return {
        "mechanism": mechanism,
        "rate": rate,
        "max_iter": max_iter,
        "tol": tol,
        "min_samples_leaf": min_samples_leaf,
        "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
        "n_iter": imputer.n_iter_,
        "seconds": round(time.perf_counter() - start, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", nargs="+", type=float, default=[0.50])
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--min-samples-leaf", nargs="+", type=int, default=[3])
    args = parser.parse_args()

    configs = CONFIGS
    if args.configs:
        configs = []
        for value in args.configs:
            max_iter, tol = value.split(",")
            configs.append((int(max_iter), float(tol)))

    df = pd.read_csv(LOCAL_DATA)
    x_complete = df[simulation.FEATURE_COLS].astype(float)
    tasks = [
        (mechanism, rate, max_iter, tol, min_samples_leaf)
        for mechanism in ["MCAR", "MAR", "MNAR"]
        for rate in args.rates
        for max_iter, tol in configs
        for min_samples_leaf in args.min_samples_leaf
    ]
    results = Parallel(n_jobs=args.workers, verbose=10)(
        delayed(test_config)(x_complete, mechanism, rate, max_iter, tol, min_samples_leaf)
        for mechanism, rate, max_iter, tol, min_samples_leaf in tasks
    )
    result_df = pd.DataFrame(results).sort_values(
        ["mechanism", "rate", "max_iter", "tol"]
    )
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
