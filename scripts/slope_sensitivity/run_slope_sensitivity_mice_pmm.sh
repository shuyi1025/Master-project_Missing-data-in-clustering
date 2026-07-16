#!/bin/bash
#PBS -N slope_mice_pmm
#PBS -l walltime=128:00:00
#PBS -l select=1:ncpus=32:mem=128gb
#PBS -o slope_mice_pmm.out
#PBS -e slope_mice_pmm.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PBS_O_WORKDIR:-$SCRIPT_DIR}"

eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate mlenv

python run_slope_sensitivity_mice_pmm.py \
  --synthetic-size all \
  --mechanisms MAR MNAR \
  --rates 0.10 0.20 0.30 0.40 0.50 \
  --slopes -0.50 -0.75 -2.00 -2.50 \
  --b 20 \
  --m 20 \
  --output-prefix sensitivity_mice_pmm_slope4
