#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=600:00:00
#SBATCH -o /home/fanyang3/github/safety-gym-curriculum/log/log_0094.out
#SBATCH -e /home/fanyang3/github/safety-gym-curriculum/log/error_0094.out
#SBATCH --partition=GPU 
#SBATCH --exclude=compute-0-[7,9,11,13,19]
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --job-name='0094_safety_gym'
# module load cuda-91 
module load singularity
#safety gym no curriculum
singularity exec  \
  /home/fanyang3/ubuntu2.simg bash -c \
  'source ~/.bashrc &&  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/fanyang3/.mujoco/mujoco200/bin && conda activate safety_gym  && \
 	python experiment.py \
  --seed='${1}' --exp_name='${2}' \
  --cpu=1 --robot=point --task=goal1 --algo=ppo_lagrangian --init_cost_lim=25 --target_cost_lim=25\
  --penalty_lr='${3}' --clip_penalty' &

wait


