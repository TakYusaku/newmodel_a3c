import os
import numpy as np
import argparse
import gym
from gym import wrappers
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.multiprocessing as mp
#from OpenGL.GL import *

from optimizer import SharedAdam
from parameter import Policy
from environment import Environment

#import logger
import test_a3c


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    os.environ["OMP_NUM_THREADS"] = "1"

    # hyperparameter の取得
    parser = argparse.ArgumentParser(description='PyTorch a3c')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--monitor', action='store_true',
                        help='save the rendered video')
    parser.add_argument('--log_dir', type=str, default='./log_dir',
                        help='save dir')
    parser.add_argument('--epoch', type=int, default=3000, metavar='N',
                        help='training epoch number') #default=10000000
    parser.add_argument('--local_t_max', type=int, default=5, metavar='N',
                        help='bias variance control parameter')
    parser.add_argument('--entropy_beta', type=float, default=0.01, metavar='E',
                        help='coefficient of entropy')
    parser.add_argument('--v_loss_coeff', type=float, default=0.5, metavar='V',
                        help='coefficient of value loss')
    parser.add_argument('--out_dim', type=int, default=128, metavar='N',
                        help='number of intermediate layer')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='L',
                        help='learning rate')#7e-4
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Environment')
    #parser.add_argument('--num_process', type=int, default=4, metavar='n',
                        #help='number of processes')
    parser.add_argument('--eps', type=float, default=0.01, metavar='E',
                        help='epsilon minimum log or std')
    parser.add_argument('--save_name', type=str, default='exp', metavar='N',
                        help='define save name')
    parser.add_argument('--save_mode', type=str, default='max', metavar='S',
                        help='save mode. all or last or max')
    parser.add_argument('--test', action='store_true',
                        help='test after training')
    parser.add_argument('--num_rollout', type=int, default=6, metavar='N',
                        help='number of rollout')
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    # 動画を保存するか
    if args.monitor:
        env = wrappers.Monitor(env, args.log_dir, force=True)
    
    '''
    #logger.add_tabular_output(os.path.join(args.log_dir, 'progress.csv'))
    #logger.log_parameters_lite(os.path.join(args.log_dir, 'params.json'), args)
    '''
    env = gym.make('CartPole-v0').unwrapped
    gbrain = Policy(env.observation_space.shape[0], env.action_space.n, out_dim=args.out_dim)#.to(device) # global brain の定義
    gbrain.share_memory()

    optimizer = SharedAdam(gbrain.parameters(), lr=args.lr, betas=(0.92, 0.999))

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    #global_ep, global_ep_r, res_queue, tr_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue(), mp.Queue()

    threads = [Environment(args, gbrain, optimizer, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [th.start() for th in threads]

    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    '''
    tr_res = []
    while True:
        r = tr_queue.get()
        if r is not None:
            tr_res.append(r)
        else:
            break
    '''
    [th.join() for th in threads]
    
    '''
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

    axL.plot(res)
    axL.set_ylabel('Moving average ep reward')
    axL.set_xlabel('Step')

    axR.plot(tr_res)
    axR.set_ylabel('total reward')
    axR.set_xlabel('Step')

    fig.savefig("./log_dir/a3c_cartpole-v1.png")
    '''
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.savefig("./log_dir/a3c_cartpole-v1_ma.png")
    
    if args.test:
        test_a3c.test(gbrain, args, env)