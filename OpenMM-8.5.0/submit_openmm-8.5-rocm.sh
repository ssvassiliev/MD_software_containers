# Usage: bash $0 <nGPUs>
sbatch << EOF
#!/bin/bash
#SBATCH -c96 --gpus-per-node=mi300a:${1}  -A def-svassili
#SBATCH --mem=32000 --time=1:0:0 --partition=debug

module load apptainer

echo $SLURM_JOB_GPUS

apptainer exec \
	--cleanenv \
	--rocm \
        --env SLURM_JOB_GPUS=\$SLURM_JOB_GPUS \
	--bind /home/svassili:/home/svassili \
	--bind /project/6033915:/project/6033915 \
	/project/6033915/svassili/ROCm-7.2-dev.sif python3 openmm_input_rocm.py

echo -n "Using ${1} GPUs, "
hostname -s
echo -n "CPU:"
cat /proc/cpuinfo | grep "model name" | uniq
EOF
