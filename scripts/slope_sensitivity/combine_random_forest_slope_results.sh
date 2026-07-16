#!/bin/bash
#PBS -N combine_slope_rf
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=10:mem=64gb
#PBS -o combine_slope_rf.out
#PBS -e combine_slope_rf.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PBS_O_WORKDIR:-$SCRIPT_DIR}"

eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate mlenv

python combine_random_forest_slope_results.py \
  --output-prefix sensitivity_random_forest_slope4
