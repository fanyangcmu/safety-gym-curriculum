#!/usr/bin/env python

from fileinput import filename
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
    parser.add_argument('--cost_lim', type=float, default=0)
    args = parser.parse_args()


    print(args.logdir)
    # import ipdb;ipdb.set_trace()

    

    cost_list = []
    worst_case_cost_list = []
    return_list = []
    epoch_list = []
    cost_lim_list = []
    data_list = []
    subfolders = os.listdir(args.logdir[0])
    for folder in subfolders:
        file_name = os.path.join(args.logdir[0], folder, 'progress.txt')
        if path.exists(file_name):
            try:
                exp_data = pd.read_table(file_name)
            except:
                print('Could not read from %s'%file_name)
            epochs =exp_data['Epoch'].to_numpy()
            returns = exp_data['AverageEpRet'].to_numpy()
            costs = exp_data['AverageEpCost'].to_numpy()
            cost_lims = exp_data['Averagecost_lim'].to_numpy()
            penalty = exp_data['Penalty'].to_numpy()
            surr_cost = exp_data['SurrCost'].to_numpy()
            loss_pi = exp_data['LossPi'].to_numpy()
            seed_data = np.stack((epochs, returns, costs, cost_lims, penalty, loss_pi, surr_cost), axis=0)
            seed_data = seed_data.transpose()
            print(seed_data.shape)
            data_list.append(seed_data)
            # epoch_list.append(epochs)
            # return_list.append(returns)
            # cost_list.append(costs)
            # cost_lim_list.append(cost_lims)
    if not os.path.exists('../results'):
        os.mkdir('../results')
    df_data_list = np.concatenate(data_list, axis=0)
    # data = [epoch_list, return_list, cost_list, cost_lim_list]
    df = pd.DataFrame(df_data_list, columns=['epoch', 'return', 'cost', 'cost_lim', 'penalty', 'loss_pi', 'surr_cost'])
    sbn.set()
    sbn.lineplot(data=df, x='epoch', y='cost')
    plt.title("Cost")
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    # plt.ylim(-10, 256)
    plt.ylim(-1, 100)
    plt.savefig(os.path.join('../results','{}_cost.png'.format(args.logdir[0].rsplit('/')[-1])))
    plt.cla()
    plt.clf()

    sbn.lineplot(data=df, x='epoch', y='return')
    plt.title("Return")
    plt.xlabel('Epoch')
    plt.ylabel('Return')
    # plt.ylim(-256, 0)
    plt.ylim(0, 30)
    plt.savefig(os.path.join('../results','{}_return.png'.format(args.logdir[0].rsplit('/')[-1])))
    plt.cla()
    plt.clf()

    for i, data in enumerate(data_list):
        epoch = data[:,0]
        cost_lim = data[:,3]
        cost = data[:,2]
        penalty = data[:,4]
        loss_pi = data[:,5]
        surr_cost = data[:,6]
        plt.plot(epoch, cost, label='Cost')
        plt.plot(epoch, cost_lim, label='Cost Limit')
        plt.plot(epoch, penalty, label='Penalty')
        plt.plot(epoch, loss_pi * 100, label='100 * Loss_pi')
        plt.plot(epoch, surr_cost * 100, label='100 * SurrCost')
        if args.cost_lim > 0:
            plt.axhline(y=args.cost_lim, color='r', linestyle='--')
        # plt.ylim(-10, 256)
        plt.ylim(-1, 100)
        plt.legend()
        plt.savefig(os.path.join('../results','{}_cost_seed_{}.png'.format(args.logdir[0].rsplit('/')[-1], i)))
        plt.cla()
        plt.clf()
    # for i, data in enumerate(data_list):
    #     epoch = data[:,0]
    #     returns = data[:,1]
    #     plt.plot(epoch, returns, label='Return')
    #     # plt.ylim(-256, 0)
    #     plt.ylim(0, 30)
    #     plt.legend()
    #     plt.savefig(os.path.join('../results','{}_return_seed_{}.png'.format(args.logdir[0].rsplit('/')[-1], i)))
    #     plt.cla()
    #     plt.clf()

