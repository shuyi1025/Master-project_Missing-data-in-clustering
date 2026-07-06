# Median/K-POD Seed/Slope Sensitivity

This folder tests whether the apparent improvement of median imputation or K-POD at high missingness is stable across different random seeds and missingness-logit slopes.

Important: `MISSINGNESS_LOGIT_SLOPE` is tested for MAR and MNAR. MAR uses other engagement variables as missingness drivers, while MNAR uses the target variable itself as the missingness driver.

Example runs:

```powershell
python scripts\sensitivity_median_seed_slope\run_median_seed_slope.py --synthetic-size 5000 --methods median kpod --mechanisms MAR MNAR --rates 0.30 0.40 0.50 --base-seeds 101 123 202 404 808 --slopes -0.50 -0.75 -1.25 -2.00 -2.50 --b 30
```

To run all three synthetic datasets in one command:

```powershell
python scripts\sensitivity_median_seed_slope\run_median_seed_slope.py --synthetic-size all --methods median kpod --mechanisms MAR MNAR --rates 0.30 0.40 0.50 --base-seeds 101 123 202 404 808 --slopes -0.50 -0.75 -1.25 -2.00 -2.50 --b 30 --output-prefix sensitivity
```

To run a stronger check over the same high-missingness region:

```powershell
python scripts\sensitivity_median_seed_slope\run_median_seed_slope.py --synthetic-size 5000 --methods median kpod --mechanisms MAR MNAR --rates 0.30 0.40 0.50 --base-seeds 101 123 202 404 808 --slopes -0.50 -0.75 -1.25 -2.00 -2.50 --b 50 --output-prefix sensitivity
```

Outputs are written to:

```text
outputs/sensitivity_median_seed_slope/
```

The detailed results file contains one row per replication/method/mechanism/rate/slope. The summary files aggregate every recorded metric, including observed missingness, ARI, centroid mean error, centroid max error, cluster-size mean absolute error, silhouette, and runtime. For each metric, the summary reports mean, standard deviation, minimum, and maximum.

On a PBS-style HPC cluster, submit all three datasets with:

```bash
qsub scripts/sensitivity_median_seed_slope/submit_all_datasets.pbs
```

This writes concise separate outputs for `n204`, `n2000`, and `n5000`, for example `sensitivity_summary_n2000.csv`, so the datasets do not overwrite each other.
