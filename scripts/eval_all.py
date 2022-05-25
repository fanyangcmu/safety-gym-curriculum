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
    parser.add_argument('--est', default='mean')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--clearx', action='store_true')
    parser.add_argument('--cost_lim', type=float, default=0)
    parser.add_argument('--paper', action='store_true', default=False)
    args = parser.parse_args()


    logdirs = [
        # '/home/fanyang3/github/safety-gym-curriculum/data/2022-05-22_0097_point_push1_curr_lr1e-1_decrease0.5_init_512_stable_10_target_d_25_clip_penalty',
        '/home/fanyang3/github/safety-gym-curriculum/data/2022-05-22_0107_point_push1_curr_lr5e-2_decrease0.5_init_512_stable_10_target_d_25',
        # '/home/fanyang3/github/safety-gym-curriculum/data/2022-05-22_0107_point_push1_curr_lr1e-2_decrease0.5_init_512_stable_10_target_d_25_clip_penalty',
        '/home/fanyang3/github/safety-gym-curriculum/data/2022-05-22_0107_point_push1_no_curr_steps_30k_penalty_lr_5e-2'
    ]
    save_fig_name = 'point_push1'
    method_list = ['SP-PPO-Lagrangian','PPO-Lagrangian']

    

    cost_list = []
    worst_case_cost_list = []
    return_list = []
    epoch_list = []
    cost_lim_list = []
    data_list = []
    for i,logdir in enumerate(logdirs):
        all_df = []
        subfolders = os.listdir(logdir)
        for folder in subfolders:
            file_name = os.path.join(logdir, folder, 'progress.txt')
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
                # if args.paper:
                #     seed_data = np.stack((epochs, returns, costs, cost_lims, method_name), axis=0)
                # else:
                seed_data = np.stack((epochs, returns, costs, cost_lims, method_id), axis=0)
                seed_data = seed_data.transpose()
                data_list.append(seed_data)
            else:
                print("warning, {} doesn't exist".format(file_name))
                # epoch_list.append(epochs)
                # return_list.append(returns)
                # cost_list.append(costs)
                # cost_lim_list.append(cost_lims)
        df_data_list = np.concatenate(data_list, axis=0)
        # data = [epoch_list, return_list, cost_list, cost_lim_list]
        df = pd.DataFrame(df_data_list, columns=['epoch', 'return', 'cost', 'cost_lim', 'method_id'])
        # if args.paper:
            # df['method_id'] = method_list[i]
        all_df.append(df)
    sbn.set()
    # for df in all_df:
    sbn.lineplot(data=df, x='epoch', y='cost', hue='method_id', palette=sbn.color_palette()[0:len(logdirs)])
    if args.cost_lim > 0:
        plt.axhline(y=args.cost_lim, color='r', linestyle='--')
    plt.title("Cost")
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.ylim(-10, 100)
    if args.paper:
        plt.legend(labels=method_list)
    # plt.ylim(-1, 10)
    if args.paper:
        plt.savefig(os.path.join('../results','{}_cost.pdf'.format(save_fig_name)))
    else:
        plt.savefig(os.path.join('../results','{}_cost.png'.format(save_fig_name)))
    # plt.show()
    plt.cla()
    plt.clf()

    # for df in all_df:
    sbn.lineplot(data=df, x='epoch', y='return', hue='method_id', palette=sbn.color_palette()[0:len(logdirs)])
    plt.title("Return")
    plt.xlabel('Epoch')
    plt.ylabel('Return')
    # plt.ylim(-300, 0)
    # plt.ylim(0, 30)
    plt.ylim(0, 10)
    if args.paper:
        plt.legend(labels=method_list)
    # plt.ylim(0, 5)
    if args.paper:
        plt.savefig(os.path.join('../results','{}_return.pdf'.format(save_fig_name)))
    else:
        plt.savefig(os.path.join('../results','{}_return.png'.format(save_fig_name)))
    plt.cla()
    plt.clf()


