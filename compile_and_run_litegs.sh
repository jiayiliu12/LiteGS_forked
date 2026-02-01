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
module list
which nvcc
nvcc --version
which gcc
gcc --version
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export C_INCLUDE_PATH="$CONDA_PREFIX/include"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include"
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
cd litegs/submodules/gaussian_raster/
ls
rm -rf build dist *.egg-info
find . -name "*.so" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
pip uninstall -y litegs_fused
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
pip install -v . --no-build-isolation --no-cache-dir
cd ~
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):$LD_LIBRARY_PATH"
python -c "import litegs_fused; print(litegs_fused.__file__)"

cd ~/LiteGS_forked/
python ./example_train.py --sh_degree 3 -s ./../data/truck/ -i images/ -m output/firsttest