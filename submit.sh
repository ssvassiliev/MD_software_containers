#!/bin/bash
#SBATCH --mem-per-cpu=4000 -c8 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --time=1:0:0 -A def-svassili

module load apptainer
apptainer exec --nv \
	../ubuntu-24.04-cuda-12.6.2-torch-2.8.sif \
	python openmm_input_ANI2X.py