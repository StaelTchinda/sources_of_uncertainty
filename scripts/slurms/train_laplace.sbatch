#!/bin/bash
#SBATCH --job-name="Stael Train"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/tchindak/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/tchindak/logs/slurm-%j.out
PROJECT_PATH=/storage/user/tchindak/sources_of_uncertainty
$PROJECT_PATH/venv/bin/python -O $PROJECT_PATH/demo/bayesian/train_laplace.py $@

# Comment (Lukas K.)
# For a normal DL task, it is useful to use multiple CPUs per GPU for example --cpus-per-task=3. This depends on how many CPUs your dataloader (pytorch num_workers) can use effectively.
# For a normal DL task, one should use about 32G of memory. Of course, this depends on the task. Try not to take too much, but if the memory is insufficient the task will get killed.