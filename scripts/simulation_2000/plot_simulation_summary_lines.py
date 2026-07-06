"""Plot mean ARI across missingness rates for every simulation method."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_DIR = PROJECT_ROOT / "outputs" / "simulation2000"
SUMMARY_PATH = SIMULATION_DIR / "simulation_summary_all_methods.csv"

METHOD_ORDER = ["knn", "kpod", "median", "mice_pmm", "random_forest"]
METHOD_LABELS = {
    "knn": "KNN",
    "kpod": "k-POD",
    "median": "Median",
    "mice_pmm": "MICE PMM",
    "random_forest": "Random Forest",
}
METHOD_COLORS = {
    "knn": "#0072B2",
    "kpod": "#E69F00",
    "median": "#009E73",
    "mice_pmm": "#CC79A7",
    "random_forest": "#D55E00",
}
METHOD_MARKERS = {
    "knn": "o",
    "kpod": "s",
    "median": "^",
    "mice_pmm": "D",
    "random_forest": "P",
}


def main() -> None:
    summary = pd.read_csv(SUMMARY_PATH)
    required_columns = {"mechanism", "target_missing_rate", "method", "mean_ari"}
    missing_columns = required_columns.difference(summary.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    for mechanism in ["MAR", "MCAR", "MNAR"]:
        mechanism_data = summary.loc[summary["mechanism"] == mechanism]
        fig, ax = plt.subplots(figsize=(8, 5))

        for method in METHOD_ORDER:
            method_data = mechanism_data.loc[
                mechanism_data["method"] == method
            ].sort_values("target_missing_rate")
            if method_data.empty:
                continue

            ax.plot(
                method_data["target_missing_rate"] * 100,
                method_data["mean_ari"],
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=2.2,
                markersize=7,
            )

        ax.set_title(f"Mean ARI by Missingness Rate ({mechanism})", fontweight="bold")
        ax.set_xlabel("Missingness Rate (%)")
        ax.set_ylabel("Mean ARI vs Full-Data K-means")
        ax.set_xticks([10, 20, 30, 40, 50], labels=["10%", "20%", "30%", "40%", "50%"])
        ax.set_xlim(8, 52)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="both", alpha=0.25, linestyle="--")
        ax.legend(title="Method", loc="lower left", frameon=True)
        fig.tight_layout()

        output_path = SIMULATION_DIR / f"mean_ari_all_methods_{mechanism.lower()}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(output_path)


if __name__ == "__main__":
    main()
