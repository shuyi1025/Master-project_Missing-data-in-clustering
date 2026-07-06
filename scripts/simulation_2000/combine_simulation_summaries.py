"""Combine all simulation summary CSV files into one consistently ordered table."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_DIR = PROJECT_ROOT / "outputs" / "simulation2000"
OUTPUT_PATH = SIMULATION_DIR / "simulation_summary_all_methods.csv"

COLUMNS = [
    "source_summary",
    "mechanism",
    "target_missing_rate",
    "method",
    "mean_ari",
    "sd_ari",
    "mean_ari_rubin_se",
    "mean_centroid_error",
    "mean_centroid_error_rubin_se",
    "mean_cluster_size_error",
    "mean_cluster_size_error_rubin_se",
    "mean_silhouette",
    "mean_silhouette_rubin_se",
    "mean_runtime_seconds",
]


def sort_key(row: dict[str, str]) -> tuple[str, float, str]:
    return (
        row["mechanism"],
        float(row["target_missing_rate"]),
        row["method"],
    )


def main() -> None:
    input_paths = sorted(
        path
        for path in SIMULATION_DIR.rglob("*summary*.csv")
        if path.resolve() != OUTPUT_PATH.resolve()
    )

    rows: list[dict[str, str]] = []
    for path in input_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as input_file:
            for row in csv.DictReader(input_file):
                row["source_summary"] = path.name
                rows.append({column: row.get(column, "") for column in COLUMNS})

    rows.sort(key=sort_key)

    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Combined {len(input_paths)} summary files and {len(rows)} rows.")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
