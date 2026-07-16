#!/bin/bash
#PBS -N slope_summary
#PBS -l walltime=128:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -o slope_summary.out
#PBS -e slope_summary.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PBS_O_WORKDIR:-$SCRIPT_DIR}"

eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate mlenv

python summarise_slope_sensitivity.py
