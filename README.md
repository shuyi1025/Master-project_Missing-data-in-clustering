# MSc Project

Python analysis scripts for exploring missing data, dataset summaries, and engagement clustering for the MSc project.

## Repository structure

- `scripts/`: analysis and plotting scripts.
- `data_raw/`: local raw data files. This folder is intentionally ignored by git.
- `outputs/`: generated tables and figures. This folder is intentionally ignored by git.

## Setup

Create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the required Stata data files in `data_raw/` before running the scripts:

- `ADSL.dta`
- `ADQS.dta`
- `COPE_Final_Indicators.dta`

## Run analyses

```powershell
python scripts\plot_cope_distributions.py
python scripts\kmeans_fully_observed_engagement.py
python scripts\simulation\simulation_gmm_kmeans.py
python scripts\simulation\run_median.py
python scripts\simulation\run_mice_pmm.py
python scripts\simulation\run_knn.py
python scripts\simulation\run_random_forest.py
python scripts\simulation\run_kpod.py
```

Generated figures and tables are written to `outputs/`.

## Data note

Raw research data and generated outputs are excluded from version control by default. Only add them to GitHub if you have confirmed that sharing them is permitted.
