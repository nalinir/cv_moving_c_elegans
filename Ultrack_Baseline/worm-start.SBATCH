#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=torch
#SBATCH --account=pr_130_general
module purge
singularity exec --nv \
        --overlay /scratch/mie8014/my_envs/wormnd2/overlay-15GB-500K.ext3:rw \
        /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif\
        /bin/bash -c "source /ext3/env.sh; python /home/rmf9265/WormND-main/neuron_tracking/ultrack/ultrack-stardist-finetuned.py /home/rmf9265/Worms"
