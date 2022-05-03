#!/bin/bash

for i in 4 5 6 7 8 9
do
    # normal experiment setup
    # sbatch seuss_sbatch_curriculum_single.sh ${i} Circle-v0 safeRL/circlev0/0058_curr_single_lr1e-1_decrease0.5_init_0_stable_10_warm_up_no_curr
    # safety gym
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0085_curr_single_lr5e-2_decrease0.5_init_0_stable_10_target_d_25
        # No curriculum
    sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0086_no_curr_steps_30k
    #Trajectory following
    # sbatch seuss_sbatch_curriculum_single.sh ${i} Trajectory-v0 safeRL/Trajectory-v0/0079_curr_lr0.1_decrease0.5_stable_20_desired_v_2_MRZR_6k_steps 2
    # sbatch seuss_sbatch_curriculum_single.sh ${i} Trajectory-v0 safeRL/Trajectory-v0/0079_curr_lr0.1_decrease0.5_stable_20_desired_v_3_MRZR_6k_steps 3
    # sbatch seuss_sbatch_curriculum_single.sh ${i} Trajectory-v0 safeRL/Trajectory-v0/0079_curr_lr0.1_decrease0.5_stable_20_desired_v_4_MRZR_6k_steps 4
        #no curriculum
    # sbatch seuss_sbatch_no_curriculum_single.sh ${i} Trajectory-v0 safeRL/Trajectory-v0/0080_no_curr_lr0.1_desired_v_2_MRZR_6k_steps 2
    # sbatch seuss_sbatch_no_curriculum_single.sh ${i} Trajectory-v0 safeRL/Trajectory-v0/0080_no_curr_lr0.1_desired_v_3_MRZR_6k_steps 3 
    # sbatch seuss_sbatch_no_curriculum_single.sh ${i} Trajectory-v0 safeRL/Trajectory-v0/0080_no_curr_lr0.1_desired_v_4_MRZR_6k_steps 4


done
