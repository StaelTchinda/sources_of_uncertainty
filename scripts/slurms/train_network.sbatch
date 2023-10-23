#!/bin/bash
#SBATCH --job-name="Stael Train MNIST - curvature retuning curvature_update"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=2G
#SBATCH --time=0:45:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/tchindak/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/tchindak/logs/slurm-%j.out
# TODO: FIX ERROR
# srun "/storage/user/tchindak/master_thesis/venv/bin/python /storage/user/tchindak/master_thesis/demo/demo.py --data MNIST --network MNIST_LENET --sampling CURVATURE_MATRIX --model_checkpoint --curvature_checkpoint --retuning --curvature_update"

# Comment (Lukas K.)
# For a normal DL task, it is useful to use multiple CPUs per GPU for example --cpus-per-task=3. This depends on how many CPUs your dataloader (pytorch num_workers) can use effectively.
# For a normal DL task, one should use about 32G of memory. Of course, this depends on the task. Try not to take too much, but if the memory is insufficient the task will get killed.