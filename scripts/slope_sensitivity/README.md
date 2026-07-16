# Imputation Slope Sensitivity

This folder runs slope sensitivity analyses for the five imputation methods:

- `median`
- `knn`
- `kpod`
- `mice_pmm`
- `random_forest`

The missingness-logit slope is tested for MAR and MNAR. MAR uses other engagement
variables as missingness drivers, while MNAR uses the target variable itself as
the missingness driver.

The default missingness rates are now 10%, 20%, 30%, 40%, and 50%.

The default base seed is fixed at `123`, because seed sensitivity has already
been ruled out. The `--base-seeds` option is still available for reproducibility
checks, but routine slope sensitivity runs should use the default single seed.

The default slope set excludes `-1.25`, because that slope is the existing main
simulation setting:

```text
-0.50, -0.75, -2.00, -2.50
```

Run each imputation method separately. Each method script runs all three
synthetic dataset sizes: n=204, n=2000, and n=5000.

The five `run_slope_sensitivity_<method>.py` files are standalone runnable
entrypoints. Each file contains the full missingness generation, imputation,
clustering, metric calculation, and summary-writing workflow for that method.

Method-level command set:

```powershell
python scripts\slope_sensitivity\run_slope_sensitivity_median.py
python scripts\slope_sensitivity\run_slope_sensitivity_knn.py
python scripts\slope_sensitivity\run_slope_sensitivity_kpod.py
python scripts\slope_sensitivity\run_slope_sensitivity_mice_pmm.py
python scripts\slope_sensitivity\run_slope_sensitivity_random_forest.py
```

For the two slowest methods, the four slopes also have separate Python/PBS
pairs. Submit these eight jobs independently so that the slopes can run in
parallel on the HPC:

```bash
qsub run_slope_sensitivity_random_forest_slope_neg050.sh
qsub run_slope_sensitivity_random_forest_slope_neg075.sh
qsub run_slope_sensitivity_random_forest_slope_neg200.sh
qsub run_slope_sensitivity_random_forest_slope_neg250.sh

qsub run_slope_sensitivity_mice_pmm_slope_neg050.sh
qsub run_slope_sensitivity_mice_pmm_slope_neg075.sh
qsub run_slope_sensitivity_mice_pmm_slope_neg200.sh
qsub run_slope_sensitivity_mice_pmm_slope_neg250.sh
```

Here `neg050`, `neg075`, `neg200`, and `neg250` mean slopes `-0.50`, `-0.75`,
`-2.00`, and `-2.50`. Each job uses a distinct output prefix, so concurrent jobs
do not overwrite one another. The summarising script automatically reads all of
the resulting seed-summary CSV files.

After the four jobs for each method have finished, combine and validate the RF
and PMM outputs separately:

```bash
qsub combine_random_forest_slope_results.sh
qsub combine_mice_pmm_slope_results.sh
```

Each combiner creates method-specific detailed, summary, and seed-summary CSVs
for every dataset size, plus three method-specific `*_all_sizes.csv` files. By
default, each script stops and reports the missing filenames unless all 12
detailed inputs for that method are present (four slopes x three dataset sizes).

For a quick smoke test, override the repetition count, rates, slopes, mechanism,
dataset size, and MICE-PMM imputation count:

```powershell
python scripts\slope_sensitivity\run_slope_sensitivity_mice_pmm.py --synthetic-size 204 --b 1 --m 2 --mechanisms MAR --rates 0.10 --slopes -0.50 --output-prefix smoke_mice_pmm_slope4
```

Outputs are written to:

```text
outputs/slope_sensitivity/
```

For each run, the script writes:

- detailed results: `<output-prefix>_results_n<size>.csv`
- grouped summary: `<output-prefix>_summary_n<size>.csv`
- seed-level summary: `<output-prefix>_summary_by_seed_n<size>.csv`

The detailed results file contains one row per
replication/method/mechanism/rate/slope. The summary files aggregate observed
missingness, ARI, centroid mean error, centroid max error, cluster-size mean
absolute error, silhouette, and runtime.

After the method-level runs finish, regenerate summary tables and figures:

```powershell
python scripts\slope_sensitivity\summarise_slope_sensitivity.py
```
