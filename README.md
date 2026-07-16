# Missing Data in K-Means Clustering of Digital Mental Health Engagement Data

This repository contains the analysis and simulation code for an MSc project at Imperial College London:

> **Evaluating Missing-Data Methods for K-Means Clustering of Digital Mental Health Engagement Data: A Simulation Study**

The project uses engagement data from the COPe-support arm of the EFFIP trial. Its purpose is to evaluate how different missing-data strategies affect recovery of K-means engagement clusters under different missingness mechanisms, missingness rates, and sample sizes.

## Research question

Digital interventions produce engagement measures such as login days, time on the platform, page views, forum posts, and use of specific website components. These measures are often incomplete, while standard K-means cannot operate directly on missing values.

This study asks which missing-data method most accurately recovers the clustering structure that would have been obtained from complete data. Five approaches are compared:

- median imputation;
- K-nearest-neighbours (KNN) imputation;
- multiple imputation by chained equations with predictive mean matching (MICE-PMM);
- random-forest imputation;
- K-POD, which alternates K-means fitting with centroid-based updates of missing entries.

## Study design

The simulation follows the ADEMP framework.

| Component | Specification |
| --- | --- |
| Aim | Compare missing-data strategies for recovery of K-means engagement clusters. |
| Data-generating mechanism | A five-component Gaussian mixture model fitted to standardised complete-case COPe engagement data. |
| Sample sizes | `n = 204`, `n = 2,000`, and `n = 5,000`. |
| Missingness mechanisms | MCAR, MAR, and MNAR. |
| Target missingness rates | 10%, 20%, 30%, 40%, and 50%. |
| Main missingness-logit slope | `-1.25`; additional slopes are examined in sensitivity analysis. |
| Methods | Median, KNN, MICE-PMM, random forest, and K-POD. |
| Monte Carlo repetitions | 500 per main simulation setting. |
| Primary performance measure | Adjusted Rand Index (ARI) against the full-data K-means benchmark. |
| Secondary measures | Mean and maximum centroid error, cluster-size error, silhouette score, realised missingness, and runtime. |

MAR missingness is driven by other observed engagement variables. MNAR missingness is driven by the value of the variable being made incomplete. The intercept of the missingness model is calibrated for each scenario so that the realised missingness is close to the target rate.

## Data

The repository includes:

```text
data_raw/COPE_Final_Indicators.dta
```

The Stata file contains 204 observations and 25 variables. It has five study or access variables and 20 engagement features:

- study/access variables: `ID`, `cohort`, `trt`, `registered`, `activated`;
- engagement features: `totalmins`, `logindays`, `loginwks`, `pageviews`, `posts`, `ptp`, `ate`, `totaldays`, `act_wkpv`, `dur_wkpv`, `act_ptp`, `dur_ptp`, `act_ate`, `dur_ate`, `sdwkpv`, `sdptp`, `sdate`, `rate_wkpv`, `rate_ptp`, and `rate_ate`.

The clustering scripts exclude participant ID and trial/access variables. Engagement variables are transformed using `log(x + 1)` and standardised before K-means or Gaussian-mixture modelling.

### Data-governance note

The dataset contains no obvious direct identifiers such as names, email addresses, telephone numbers, postal addresses, or dates of birth. However, `ID` is a unique participant-level study identifier. Access, redistribution, and reuse must remain consistent with the EFFIP/COPe-support data-sharing permissions, research approvals, and institutional policy. Inclusion in this repository should not be interpreted as a general-purpose open-data licence.

## Repository structure

```text
.
|-- data_raw/                       # Source COPe engagement data
|-- outputs/                        # Generated tables, figures, and simulation results
|   |-- cope_distributions/         # Exploratory data-analysis outputs
|   |-- kmeans_fully_observed_engagement_k5/
|   |-- simulation_gmm_kmeans204/   # Synthetic complete datasets and benchmarks
|   |-- simulation_gmm_kmeans2000/
|   |-- simulation_gmm_kmeans5000/
|   |-- simulation204/              # Missing-data simulation summaries
|   |-- simulation2000/
|   |-- simulation5000/
|   `-- slope_sensitivity/
|-- scripts/
|   |-- plot_cope_distributions.py
|   |-- kmeans_fully_observed_engagement.py
|   |-- simulation_204/             # n=204 generation, methods, summaries, plots
|   |-- simulation_2000/            # n=2,000 generation, methods, summaries, plots
|   |-- simulation_5000/            # n=5,000 generation, methods, summaries, plots
|   `-- slope_sensitivity/           # Cross-size MAR/MNAR slope sensitivity workflow
|-- requirements.txt
`-- README.md
```

Generated outputs are ignored by default so that rerunning a large experiment does not accidentally add many files. Results that were already tracked remain in version control. Raw data are intentionally included for this repository's reproducibility workflow, subject to the data-governance note above.

## Installation

Python 3.10 or later is recommended.

```powershell
git clone https://github.com/shuyi1025/Master-project_Missing-data-in-clustering.git
cd Master-project_Missing-data-in-clustering

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Analysis workflow

Run commands from the repository root. All Python paths are resolved relative to the repository, so the same commands work locally and on an HPC system after the repository has been copied or cloned.

### 1. Explore the observed data

```powershell
python scripts\plot_cope_distributions.py
```

This produces distribution plots, missingness summaries, categorical counts, and grouped boxplots in `outputs/cope_distributions/`.

### 2. Fit the complete-case K-means reference analysis

```powershell
python scripts\kmeans_fully_observed_engagement.py
```

The script evaluates candidate values of `k` from 2 to 8 and currently uses `k = 5` for the reference clustering. It writes cluster assignments, standardised and original-scale profiles, PCA results, diagnostics, and figures to `outputs/kmeans_fully_observed_engagement_k5/`.

### 3. Generate complete synthetic datasets

```powershell
python scripts\simulation_204\simulation_gmm_kmeans_204.py
python scripts\simulation_2000\simulation_gmm_kmeans_2000.py
python scripts\simulation_5000\simulation_gmm_kmeans_5000.py
```

Each script fits or reuses a five-component Gaussian mixture model and writes a synthetic complete dataset plus its full-data K-means benchmark to the matching `outputs/simulation_gmm_kmeans<size>/` directory.

### 4. Run the main simulation at n=204

```powershell
python scripts\simulation_204\run_median_204.py
python scripts\simulation_204\run_knn_204.py
python scripts\simulation_204\run_kpod_204.py
python scripts\simulation_204\run_mice_pmm_204.py
python scripts\simulation_204\run_random_forest_204.py
```

### 5. Run the main simulation at n=2,000

```powershell
python scripts\simulation_2000\run_median.py
python scripts\simulation_2000\run_knn.py
python scripts\simulation_2000\run_kpod.py
python scripts\simulation_2000\run_mice_pmm.py
python scripts\simulation_2000\run_random_forest.py
```
### 6. Run the main simulation at n=5,000

```powershell
python scripts\simulation_5000\run_median.py
python scripts\simulation_5000\run_knn.py
python scripts\simulation_5000\run_kpod.py
python scripts\simulation_5000\run_mice_pmm.py
python scripts\simulation_5000\run_random_forest.py
```

Use `python <script> --help` to inspect supported overrides such as repetition count, missingness mechanism, rate, and number of imputations. Full runs are computationally expensive; use small overrides for smoke tests before submitting 500-repetition jobs.

### 7. Combine summaries and draw comparison plots

For `n=204`:

```powershell
python scripts\simulation_204\combine_simulation_summaries_204.py
python scripts\simulation_204\plot_simulation_summary_lines_204.py
python scripts\simulation_204\plot_simulation_silhouette_lines_204.py
```

For `n=2,000`:

```powershell
python scripts\simulation_2000\combine_simulation_summaries.py
python scripts\simulation_2000\plot_simulation_summary_lines.py
python scripts\simulation_2000\plot_simulation_silhouette_lines.py
```

For `n=5,000`:

```powershell
python scripts\simulation_5000\combine_simulation_summaries_5000.py
python scripts\simulation_5000\plot_simulation_summary_lines_5000.py
python scripts\simulation_5000\plot_simulation_silhouette_lines_5000.py
```

## Slope sensitivity analysis

The slope-sensitivity workflow varies the MAR/MNAR logistic slope over `-0.50`, `-0.75`, `-2.00`, and `-2.50`; `-1.25` is the main simulation value. Each method-level runner supports all three synthetic sample sizes.

```powershell
python scripts\slope_sensitivity\run_slope_sensitivity_median.py
python scripts\slope_sensitivity\run_slope_sensitivity_knn.py
python scripts\slope_sensitivity\run_slope_sensitivity_kpod.py
python scripts\slope_sensitivity\run_slope_sensitivity_mice_pmm.py
python scripts\slope_sensitivity\run_slope_sensitivity_random_forest.py
```

A small smoke test can be run with:

```powershell
python scripts\slope_sensitivity\run_slope_sensitivity_mice_pmm.py --synthetic-size 204 --b 1 --m 2 --mechanisms MAR --rates 0.10 --slopes -0.50 --output-prefix smoke_mice_pmm
```

After all method runs finish:

```powershell
python scripts\slope_sensitivity\summarise_slope_sensitivity.py
```

See [`scripts/slope_sensitivity/README.md`](scripts/slope_sensitivity/README.md) for output naming and PBS job details.

## HPC use

PBS submission scripts are provided in `scripts/slope_sensitivity/`. They use `PBS_O_WORKDIR` when submitted with `qsub` and otherwise fall back to their own directory, so no user-specific absolute project path is required.

Submit jobs from the slope-sensitivity directory, for example:

```bash
cd scripts/slope_sensitivity
qsub run_slope_sensitivity_median.sh
```

Review the requested wall time, CPU count, memory, Conda environment name, and repetition count before submission because these settings are cluster-specific.


## Project status

This is an active MSc research repository. The analysis code, simulation settings, results, and manuscript are still being refined; values in the final submitted report should take precedence over preliminary repository summaries.
