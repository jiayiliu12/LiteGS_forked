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
# rm -rf ~/wandb_sweep_copy/
# cp -R ~/LiteGS_forked/ ~/wandb_sweep_copy
module load gcc/10.2.0
module load cuda/12.1.1

python ~/wandb_sweep_copy/example_train.py --sh_degree 3 -s ./../data/truck/ -i images/ -m output/firsttest
# python ./example_train.py --sh_degree 3 -s ./../data/truck/ -i images/ -m output/firsttest --soft_prune_epoch_interval 10x