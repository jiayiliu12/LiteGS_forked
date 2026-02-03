#!/bin/bash
#SBATCH -A g34
#SBATCH -J litegs
#SBATCH --nodelist=ault25
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o litegs_%j.out
#SBATCH -e litegs_%j.err

source /users/ljiayi/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate litegs
cd ~/LiteGS_forked/
module load gcc/10.2.0
module load cuda/12.1.1

python ./example_train_wandbsweep.py --sh_degree 3 -s ./../data/truck/ -i images/ -m output/firsttest