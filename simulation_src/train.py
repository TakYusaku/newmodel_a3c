import os
import numpy as np
import argparse
import gym
from gym import wrappers
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
import logger
import util
import test


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
    parser.add_argument('--epoch', type=int, default=2000, metavar='N',
                        help='training epoch number') #default=10000000
    parser.add_argument('--local_t_max', type=int, default=5, metavar='N',
                        help='bias variance control parameter')
    parser.add_argument('--entropy_beta', type=float, default=0.01, metavar='E',
                        help='coefficient of entropy')
    parser.add_argument('--c_loss_coeff', type=float, default=0.5, metavar='V',
                        help='coefficient of value loss')
    parser.add_argument('--out_dim', type=int, default=128, metavar='N',
                        help='number of intermediate layer')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='L',
                        help='learning rate')#7e-4
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Environment')
    #parser.add_argument('--num_process', type=int, default=4, metavar='n',
                        #help='number of processes')
    parser.add_argument('--num_process', action='store_true',
                        help='dont use mp.cpu_count()')
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
    
    env = gym.make('CartPole-v0').unwrapped
    env = util.init_setting(env, args)

    gbrain = Policy(env.observation_space.shape[0], env.action_space.n, out_dim=args.out_dim)
    gbrain.share_memory()

    optimizer = SharedAdam(gbrain.parameters(), lr=args.lr, betas=(0.92, 0.999))

    #global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    global_ep, global_ep_r, res_queue, tr_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue(), mp.Queue()

    if args.num_process:
        num_process = 4
    else:
        num_process = mp.cpu_count()

    #threads = [Environment(args, gbrain, optimizer, global_ep, global_ep_r, res_queue, i) for i in range(num_process)]
    threads = [Environment(args, gbrain, optimizer, global_ep, global_ep_r, res_queue, tr_queue, i) for i in range(num_process)]

    [th.start() for th in threads]

    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    
    tr_res = []
    while True:
        r = tr_queue.get()
        if r is not None:
            tr_res.append(r)
        else:
            break
    
    [th.join() for th in threads]

    util.plot_graph(res, tr_res)

    print('###########################################')
    print('Learning completed')
    print('###########################################')

    if args.test:
        if args.save_mode == 'max':
            tbrain = Policy(env.observation_space.shape[0], env.action_space.n, out_dim=args.out_dim)
            tbrain.load_state_dict(torch.load(os.path.join(util.nwparam_dirname, args.save_name+"_max_{}.pkl".format(int(max(tr_res))))))
        threads[0].test(tbrain, env, args)