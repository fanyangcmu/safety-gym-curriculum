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
    parser.add_argument('--cost_lim', type=float, default=0.1)
    args = parser.parse_args()


    print(args.logdir)
    # import ipdb;ipdb.set_trace()

    

    cost_list = []
    worst_case_cost_list = []
    return_list = []
    epoch_list = []
    cost_lim_list = []
    data_list = []
    for seed in range(4,10):
        file_name=  args.logdir[0]+'_s'+str(seed)+"/progress.txt"
        if path.exists(file_name):
            try:
                exp_data = pd.read_table(file_name)
            except:
                print('Could not read from %s'%file_name)
            epochs =exp_data['Epoch'].to_numpy()
            returns = exp_data['AverageEpRet'].to_numpy()
            costs = exp_data['AverageEpCost'].to_numpy()
            cost_lims = exp_data['Averagecost_lim'].to_numpy()
            seed_data = np.stack((epochs, returns, costs, cost_lims), axis=0)
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
    df = pd.DataFrame(df_data_list, columns=['epoch', 'return', 'cost', 'cost_lim'])
    sbn.set()
    sbn.lineplot(data=df, x='epoch', y='cost')
    plt.title("Cost")
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    # plt.ylim(-10, 256)
    plt.ylim(-1, 100)
    plt.savefig(os.path.join('../results','{}_cost.jpg'.format(args.logdir[0].rsplit('/')[-1])))
    plt.cla()
    plt.clf()

    sbn.lineplot(data=df, x='epoch', y='return')
    plt.title("Return")
    plt.xlabel('Epoch')
    plt.ylabel('Return')
    # plt.ylim(-256, 0)
    plt.ylim(0, 30)
    plt.savefig(os.path.join('../results','{}_return.jpg'.format(args.logdir[0].rsplit('/')[-1])))
    plt.cla()
    plt.clf()

    for i, data in enumerate(data_list):
        epoch = data[:,0]
        cost_lim = data[:,3]
        cost = data[:,2]
        plt.plot(epoch, cost, label='Cost')
        plt.plot(epoch, cost_lim, label='Cost Limit')
        # plt.ylim(-10, 256)
        plt.ylim(-1, 100)
        plt.legend()
        plt.savefig(os.path.join('../results','{}_cost_seed_{}.jpg'.format(args.logdir[0].rsplit('/')[-1], i)))
        plt.cla()
        plt.clf()
    for i, data in enumerate(data_list):
        epoch = data[:,0]
        returns = data[:,1]
        plt.plot(epoch, returns, label='Cost')
        # plt.ylim(-256, 0)
        plt.ylim(0, 30)
        plt.legend()
        plt.savefig(os.path.join('../results','{}_return_seed_{}.jpg'.format(args.logdir[0].rsplit('/')[-1], i)))
        plt.cla()
        plt.clf()




        # for epoch_add in tqdm.tqdm(range(args.start_epoch, args.end_epoch)):

        #     file_name=  args.logdir[0]+'_s'+str(seed)+"/eval_epoch_"+str(epoch_add)+".npy"
        #     if path.exists(file_name):
                
        #         eval = np.load(file_name,allow_pickle=True).item()

        #         phi_values = np.array(eval['phi_values'])
        #         costs = np.array(eval['cost'])
        #         # if costs.max() > 1:
        #         #     costs[0] += 100000
        #         if costs.max() <= args.cost_lim:
        #             valid_epoch_index_list.append(epoch_add - args.start_epoch)
        #         returns = np.array(eval['return'])
        #         top_percent = 0.1

        #         worst_case_costs.append(np.mean(np.sort(costs)[-int(top_percent*phi_values.shape[0]):]))
        #         if args.max_cost:
        #             cost_stats.append(np.max(costs))
        #         else:
        #             cost_stats.append(np.mean(costs))
        #         mean_returns.append(returns[-1])
        #         std_costs.append(np.std(costs))
        #         # std_returns.append(np.std(returns))

        #         if args.max_cost:
        #             seed_cost.append(np.max(costs))
        #         else:                    
        #             seed_cost.append(np.mean(costs))
        #         seed_return.append(returns[-1])
        #         seed_cost_std.append(np.std(costs))
        #         # seed_return_std.append(np.std(returns))
        #         seed_worst_case_cost.append(np.mean(np.sort(costs)[-int(top_percent*phi_values.shape[0]):]))
        # seed_valid_epoch_list.append(valid_epoch_index_list)
        # if seed_cost:
        #     seed_cost_list.append(seed_cost)
        #     seed_return_list.append(seed_return)
        #     seed_cost_std_list.append(seed_cost_std)
        #     # seed_return_std_list.append(seed_return_std)
        #     seed_worst_case_cost_list.append(seed_worst_case_cost)
        #     seed_list.append(seed)
    # worst_case_costs = np.array(worst_case_costs)
    # cost_stats= np.array(cost_stats)
    # mean_returns=np.array(mean_returns)
    # std_costs = np.array(std_costs)
    # # std_returns = np.array(std_returns)
    # seed_cost_list = np.array(seed_cost_list)
    # seed_return_list = np.array(seed_return_list)
    # seed_cost_std_list = np.array(seed_cost_std_list)
    # # seed_return_std_list = np.array(seed_return_std_list)
    # seed_worst_case_cost_list = np.array(seed_worst_case_cost_list)
    # "select the highest reward among the valid epochs"
    # np.set_printoptions(precision=3)
    # num_seed_used = int(np.floor(seed_cost_list.shape[0] * args.top_ratio))
    # best_seed_list = []
    # best_index_list = []
    # best_reward_list = []
    # # seed_valid_epoch_list = np.array(seed_valid_epoch_list) / 10
    # # seed_valid_epoch_list = np.arange(len(seed_valid_epoch_list))
    # for i, valid_epoch_index_list in enumerate(seed_valid_epoch_list):
    #     if valid_epoch_index_list:
    #         valid_return_list = seed_return_list[i, valid_epoch_index_list]
    #         best_reward = np.array(valid_return_list).max()
    #         best_reward_index = np.argsort(-np.array(valid_return_list))[0]
    #         best_reward_index = valid_epoch_index_list[best_reward_index]
    #         best_seed_list.append(i)
    #         best_index_list.append(best_reward_index)
    #         best_reward_list.append(best_reward)

    # # best_seed_list = []
    # # best_index_list = []
    # # best_reward_list = []
    # # for i, seed_cost in enumerate(seed_cost_list):
    # #     zero_cost_index, = np.where(seed_cost <= 1)
    # #     if len(zero_cost_index) > 0:
    # #         best_reward = seed_return_list[i, zero_cost_index].max()
    # #         best_reward_index = np.argsort(-seed_return_list[i, zero_cost_index])[0]
    # #         best_reward_index = zero_cost_index[best_reward_index]
    # #         best_seed_list.append(i)
    # #         best_index_list.append(best_reward_index)
    # #         best_reward_list.append(best_reward)
    # return_seed_rank = np.argsort(-np.array(best_reward_list))
    # selected_seed = np.array(best_seed_list)[return_seed_rank[:num_seed_used]]
    # selected_index = np.array(best_index_list)[return_seed_rank[:num_seed_used]]

    # all_fail_flag = False

    # print("selected seed:", selected_seed + 4)
    # print("selected epoch index", selected_index + args.start_epoch)
    # if len(selected_seed) == 0:
    #     all_fail_flag = True
    # if all_fail_flag == False:
    #     print("Return mean :{:03f}".format(np.mean(seed_return_list[selected_seed, selected_index].mean())))
    #     # print("Return Std :{:03f}".format(seed_return_std_list[selected_seed, selected_index].mean()))
    #     if args.max_cost:
    #         print("Cost max :{:03f}".format(np.mean(seed_cost_list[selected_seed, selected_index].mean())))
    #     else:
    #         print("Cost mean :{:03f}".format(np.mean(seed_cost_list[selected_seed, selected_index].mean())))
    #     print("Cost std :{:03f}".format(seed_cost_std_list[selected_seed, selected_index].mean()))
    #     print("Top 10% mean :{:03f}".format(seed_worst_case_cost_list[selected_seed, selected_index].mean()))
    #     print("Mean cost of each seed")
    #     print(seed_cost_list.mean(axis=1))

    #     mean_return = seed_return_list.mean(axis=0)
    #     mean_cost = np.array(seed_cost_list).mean(axis=0)
    #     print("Mean return of all {}".format(np.mean(mean_return)))
    #     print("Mean cost of all {}".format(np.mean(cost_stats)))
        

    # "plotting "
    # for i in range(len(seed_cost_list)):
    #     plt.plot(np.arange(args.start_epoch, args.end_epoch), seed_cost_list[i], label="seed {}".format(seed_list[i]))
    # if all_fail_flag == False:
    #     plt.plot(np.arange(args.start_epoch, args.end_epoch), mean_cost, label='mean', ls='--')
    #     for i, seed in enumerate(selected_seed):
    #         plt.scatter(selected_index[i], seed_cost_list[seed,selected_index[i]], s=100, c='r')


    
    # plt.title(args.logdir[0].rsplit('/')[-1] + ' cost')
    # plt.ylim([-0.1, 30])
    # # plt.ylim([-0.1, 100])
    # plt.legend()
    # plt.ylabel('Cost')
    # plt.xlabel('Epoch')
    # if not os.path.exists('results'):
    #     os.system('mkdir results')
    # print(os.path.join('results','{}_cost.jpg'.format(args.logdir[0].rsplit('/')[-1])))

    # plt.clf()
    # plt.cla()
    # mean_return = seed_return_list.mean(axis=0)
    # for i in range(len(seed_cost_list)):
    #     plt.plot(np.arange(args.start_epoch, args.end_epoch), seed_return_list[i], label="seed {}".format(seed_list[i]))
    # if all_fail_flag == False:
    #     plt.plot(np.arange(args.start_epoch, args.end_epoch), mean_return, label='mean', ls='--')
    #     for i, seed in enumerate(selected_seed):
    #         plt.scatter(selected_index[i], seed_return_list[seed, selected_index[i]], s=100, c='r')
    # plt.title(args.logdir[0].rsplit('/')[-1] + ' return')
    # plt.ylim([-300, 0])
    # plt.legend()
    # plt.ylabel('Return')
    # plt.xlabel('Epoch')
    # print(os.path.join('results','{}_return.jpg'.format(args.logdir[0].rsplit('/')[-1])))
    # plt.savefig(os.path.join('results','{}_return.jpg'.format(args.logdir[0].rsplit('/')[-1])))
