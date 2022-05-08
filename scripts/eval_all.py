#!/usr/bin/env python

from re import L
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as patches
from os import path
import os

import pandas as pd
import tqdm

import seaborn as sbn
def find_mean_std(data):
    data = np.array(data)
    # print(data.shape)

    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return mean,std

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', nargs='*', default='TotalEnvInteracts',
                        help="example: ./model_free/data/safeRL/trajectoryv0/adversarial_1k_samples/safeRL/trajectoryv0/adversarial_1k_samples")
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    parser.add_argument('--paper', action='store_true')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--dont_show', action='store_true')
    parser.add_argument('--clearx', action='store_true')
    parser.add_argument('--top_ratio', type=float, default=1,
                        help='we only compare the top this part of the results')
    parser.add_argument('--start_epoch', type=int, default=100)
    parser.add_argument('--end_epoch', type=int, default=120)
    parser.add_argument('--cost_lim', type=float, default=0.1)
    args = parser.parse_args()


    logdirs = [
    '/home/fanyang3/github/safety-gym-curriculum/data/2022-05-03_0086_no_curr_steps_30k/2022-05-03_17-24-28-0086_no_curr_steps_30k',
    # '/home/fanyang3/github/safety-gym-curriculum/data/2022-05-05_0093_curr_lr1e-1_decrease0.5_init_512_stable_10_target_d_25/2022-05-05_18-48-25-0093_curr_lr1e-1_decrease0.5_init_512_stable_10_target_d_25',
    # '/home/fanyang3/github/safety-gym-curriculum/data/2022-05-05_0093_curr_lr1e-2_decrease0.5_init_512_stable_10_target_d_25/2022-05-05_18-48-25-0093_curr_lr1e-2_decrease0.5_init_512_stable_10_target_d_25',
    # '/home/fanyang3/github/safety-gym-curriculum/data/2022-05-05_0093_curr_lr5e-2_decrease0.5_init_512_stable_10_target_d_25/2022-05-05_18-48-25-0093_curr_lr5e-2_decrease0.5_init_512_stable_10_target_d_25',
    # '/home/fanyang3/github/safety-gym-curriculum/data/2022-05-05_0094_curr_lr1e-1_decrease0.5_init_512_stable_10_target_d_25_clip_penalty/2022-05-05_18-47-13-0094_curr_lr1e-1_decrease0.5_init_512_stable_10_target_d_25_clip_penalty',
    # '/home/fanyang3/github/safety-gym-curriculum/data/2022-05-05_0094_curr_lr1e-2_decrease0.5_init_512_stable_10_target_d_25_clip_penalty/2022-05-05_18-47-13-0094_curr_lr1e-2_decrease0.5_init_512_stable_10_target_d_25_clip_penalty',
    # '/home/fanyang3/github/safety-gym-curriculum/data/2022-05-05_0094_curr_lr5e-2_decrease0.5_init_512_stable_10_target_d_25_clip_penalty/2022-05-05_18-47-13-0094_curr_lr5e-2_decrease0.5_init_512_stable_10_target_d_25_clip_penalty',
    '/home/fanyang3/github/safety-gym-curriculum/data/2022-05-05_0094_no_curr_steps_30k_penalty_lr_1e-1_clip_penalty/2022-05-05_18-47-13-0094_no_curr_steps_30k_penalty_lr_1e-1_clip_penalty',
    '/home/fanyang3/github/safety-gym-curriculum/data/2022-05-05_0094_no_curr_steps_30k_penalty_lr_1e-2_clip_penalty/2022-05-05_18-47-13-0094_no_curr_steps_30k_penalty_lr_1e-2_clip_penalty',
    '/home/fanyang3/github/safety-gym-curriculum/data/2022-05-05_0094_no_curr_steps_30k_penalty_lr_5e-2_clip_penalty/2022-05-05_18-47-13-0094_no_curr_steps_30k_penalty_lr_5e-2_clip_penalty',
    ]    
    save_fig_name = 'safety_gym_clip_penalty'

    

    cost_list = []
    worst_case_cost_list = []
    return_list = []
    epoch_list = []
    cost_lim_list = []
    data_list = []
    for i,logdir in enumerate(logdirs):
        all_df = []
        for seed in range(4,10):
            file_name=  logdirs[i]+'_s'+str(seed)+"/progress.txt"
            if path.exists(file_name):
                try:
                    exp_data = pd.read_table(file_name)
                except:
                    print('Could not read from %s'%file_name)
                epochs =exp_data['Epoch'].to_numpy()
                returns = exp_data['AverageEpRet'].to_numpy()
                costs = exp_data['AverageEpCost'].to_numpy()
                cost_lims = exp_data['Averagecost_lim'].to_numpy()
                method_id = np.tile(i, len(epochs))
                seed_data = np.stack((epochs, returns, costs, cost_lims, method_id), axis=0)
                seed_data = seed_data.transpose()
                data_list.append(seed_data)
                # epoch_list.append(epochs)
                # return_list.append(returns)
                # cost_list.append(costs)
                # cost_lim_list.append(cost_lims)
        df_data_list = np.concatenate(data_list, axis=0)
        # data = [epoch_list, return_list, cost_list, cost_lim_list]
        df = pd.DataFrame(df_data_list, columns=['epoch', 'return', 'cost', 'cost_lim', 'method_id'])
        all_df.append(df)
    # for i in range(len(all_df)):
    #     import pdb
    #     pdb.set_trace()
    #     all_df['method_id'] = np.tile(i, 10)
    sbn.set()
    # for df in all_df:
    sbn.lineplot(data=df, x='epoch', y='cost', hue='method_id', palette=sbn.color_palette()[0:len(logdirs)])
    plt.title("Cost")
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.ylim(-10, 100)
    # plt.ylim(-1, 10)
    plt.savefig(os.path.join('../results','{}_cost.jpg'.format(save_fig_name)))
    # plt.show()
    plt.cla()
    plt.clf()

    # for df in all_df:
    sbn.lineplot(data=df, x='epoch', y='return', hue='method_id', palette=sbn.color_palette()[0:len(logdirs)])
    plt.title("Return")
    plt.xlabel('Epoch')
    plt.ylabel('Return')
    # plt.ylim(-300, 0)
    plt.ylim(0, 30)
    plt.savefig(os.path.join('../results','{}_return.jpg'.format(save_fig_name)))
    plt.cla()
    plt.clf()


