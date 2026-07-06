"""Generate a 5,000-observation synthetic COPE engagement dataset."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "simulation_204"))

import simulation_gmm_kmeans_204 as base


base.N_SYNTHETIC = 5000
base.OUT_DIR = base.ROOT / "outputs" / "simulation_gmm_kmeans5000"


if __name__ == "__main__":
    base.main()
