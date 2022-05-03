#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=600:00:00
#SBATCH -o /home/fanyang3/github/lipschitz_sampling/log/log_0085.out
#SBATCH -e /home/fanyang3/github/lipschitz_sampling/log/error_0085.out
#SBATCH --partition=GPU 
#SBATCH --exclude=compute-0-[7,9,11,13,19]
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --job-name='0085_safety_gym'
# module load cuda-91 
module load singularity

singularity exec  \
  /home/fanyang3/ubuntu2.simg bash -c \
  'source ~/.bashrc &&  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/fanyang3/.mujoco/mujoco200/bin && conda activate safety_gym  && \
 	python experiment.py \
  --seed='${1}' --exp_name='${2}' \
  --cpu=1 --robot=point --task=goal1 --algo=ppo_lagrangian --curriculum --init_cost_lim=25 --target_cost_lim=25 --decrease_ratio=0.5 --stable_length=10' &

# PYTHONPATH=$PYTHONPATH:/home/fanyang3/github/lipschitz_sampling python model_free/ppo_transfer_curriculum_single.py \ 
#    --seed=6 --env=Safexp-PointGoal1-v0 --steps=30000 --exp_name=9999test \
#      --cpu=1 --save_checkpoint --save_freq=5  \
#      --penalty_lr=0.05 --init_cost_lim=1024 --stable_length=10 --decrease_ratio=0.5\
#      --penalty_init=0
wait


