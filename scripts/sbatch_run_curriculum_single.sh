#!/bin/bash

for i in 4 5 6 7 8
do
    # normal experiment setup
    # sbatch seuss_sbatch_curriculum_single.sh ${i} Circle-v0 safeRL/circlev0/0058_curr_single_lr1e-1_decrease0.5_init_0_stable_10_warm_up_no_curr
    #goal
    # safety gym
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_goal1_curr_lr5e-2_decrease0.5_init_512_stable_10_target_d_25 goal1 0.05
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_goal1_curr_lr1e-2_decrease0.5_init_512_stable_10_target_d_25_clip_penalty goal1 0.01
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_goal1_curr_lr1e-1_decrease0.5_init_512_stable_10_target_d_25_clip_penalty goal1 0.1
    # No curriculum
    sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_goal1_no_curr_steps_30k_penalty_lr_5e-2 goal1 0.05
    # sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_goal1_no_curr_steps_30k_penalty_lr_1e-2 goal1 0.01
    # sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_goal1_no_curr_steps_30k_penalty_lr_1e-1 goal1 0.1
    
    # safety gym
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_goal2_curr_lr5e-2_decrease0.5_init_512_stable_10_target_d_25 goal2 0.05
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_goal2_curr_lr1e-2_decrease0.5_init_512_stable_10_target_d_25_clip_penalty goal2 0.01
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_goal2_curr_lr1e-1_decrease0.5_init_512_stable_10_target_d_25_clip_penalty goal2 0.1
    # No curriculum
    sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_goal2_no_curr_steps_30k_penalty_lr_5e-2 goal2 0.05
    # sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_goal2_no_curr_steps_30k_penalty_lr_1e-2 goal2 0.01
    # sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_goal2_no_curr_steps_30k_penalty_lr_1e-1 goal2 0.1

    #push
    # safety gym
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_push1_curr_lr5e-2_decrease0.5_init_512_stable_10_target_d_25 push1 0.05
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_push1_curr_lr1e-2_decrease0.5_init_512_stable_10_target_d_25_clip_penalty push1 0.01
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_push1_curr_lr1e-1_decrease0.5_init_512_stable_10_target_d_25_clip_penalty push1 0.1
    # No curriculum
    sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_push1_no_curr_steps_30k_penalty_lr_5e-2 push1 0.05
    # sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_push1_no_curr_steps_30k_penalty_lr_1e-2 push1 0.01
    # sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_push1_no_curr_steps_30k_penalty_lr_1e-1 push1 0.1
    
    # safety gym
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_push2_curr_lr5e-2_decrease0.5_init_512_stable_10_target_d_25 push2 0.05
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_push2_curr_lr1e-2_decrease0.5_init_512_stable_10_target_d_25_clip_penalty push2 0.01
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_push2_curr_lr1e-1_decrease0.5_init_512_stable_10_target_d_25_clip_penalty push2 0.1
    # No curriculum
    sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_push2_no_curr_steps_30k_penalty_lr_5e-2 push2 0.05
    # sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_push2_no_curr_steps_30k_penalty_lr_1e-2 push2 0.01
    # sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_push2_no_curr_steps_30k_penalty_lr_1e-1 push2 0.1

    #button
    # safety gym
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_button1_curr_lr5e-2_decrease0.5_init_512_stable_10_target_d_25 button1 0.05
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_button1_curr_lr1e-2_decrease0.5_init_512_stable_10_target_d_25_clip_penalty button1 0.01
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_button1_curr_lr1e-1_decrease0.5_init_512_stable_10_target_d_25_clip_penalty button1 0.1
    # No curriculum
    sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_button1_no_curr_steps_30k_penalty_lr_5e-2 button1 0.05
    # sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_button1_no_curr_steps_30k_penalty_lr_1e-2 button1 0.01
    # sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_button1_no_curr_steps_30k_penalty_lr_1e-1 button1 0.1
    
    # safety gym
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_button2_curr_lr5e-2_decrease0.5_init_512_stable_10_target_d_25 button2 0.05
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_button2_curr_lr1e-2_decrease0.5_init_512_stable_10_target_d_25_clip_penalty button2 0.01
    sbatch seuss_sbatch_curriculum_single.sh ${i} 0107_point_button2_curr_lr1e-1_decrease0.5_init_512_stable_10_target_d_25_clip_penalty button2 0.1
    # No curriculum
    sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_button2_no_curr_steps_30k_penalty_lr_5e-2 button2 0.05
    # sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_button2_no_curr_steps_30k_penalty_lr_1e-2 button2 0.01
    # sbatch seuss_sbatch_no_curriculum_single.sh ${i} 0107_point_button2_no_curr_steps_30k_penalty_lr_1e-1 button2 0.1


done
