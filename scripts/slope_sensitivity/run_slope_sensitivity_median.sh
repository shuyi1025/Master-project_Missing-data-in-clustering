#!/bin/bash
#PBS -N slope_median
#PBS -l walltime=128:00:00
#PBS -l select=1:ncpus=32:mem=128gb
#PBS -o slope_median.out
#PBS -e slope_median.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PBS_O_WORKDIR:-$SCRIPT_DIR}"

eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate mlenv

python run_slope_sensitivity_median.py \
  --synthetic-size all \
  --mechanisms MAR MNAR \
  --rates 0.10 0.20 0.30 0.40 0.50 \
  --slopes -0.50 -0.75 -2.00 -2.50 \
  --b 20 \
  --output-prefix sensitivity_median_slope4
