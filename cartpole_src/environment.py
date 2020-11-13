import os
import numpy as np
import gym
from gym import wrappers
import time
from collections import deque
import torch
import torch.autograd as autograd
import torch.multiprocessing as mp
from torch.autograd import Variable

from parameter import Policy
import util


class Environment(mp.Process):
    def __init__(self, args, gbrain, optimizer, global_ep, global_ep_r, res_queue, tr_queue, name):
        super(Environment, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue, self.tr_queue = global_ep, global_ep_r, res_queue, tr_queue
        self.gbrain, self.optimizer = gbrain, optimizer
        self.env = gym.make('CartPole-v0').unwrapped
        self.lbrain = Policy(self.env.observation_space.shape[0], self.env.action_space.n)           # local network
        self.args = args

    def run(self):
        step = 1
        o = self.env.reset()
        ep_r = 0.
        self.max_ep_r = 0.
        entropies = []
        vs = []
        while self.g_ep.value < self.args.epoch:
            observations, actions, values, rewards = [], [], [], []
            done = False

            while True:
                '''
                if self.name == 'w00':
                    self.env.render()
                '''
                a, _, v = self.lbrain.get_action(util.v_wrap(o[None, :]))
                o_, r, done, _ = self.env.step(a)
                if done: r = -1
                ep_r += r
                observations.append(o)
                actions.append(a)
                rewards.append(r)
                values.append(v)


                step += 1

                if step % self.args.local_t_max == 0 or done:  # update global and assign to local net
                    v_loss, entropy = util.updating(self.args, self.gbrain, self.lbrain, self.optimizer, o_, done, observations, actions, rewards)
                    vs.append(v_loss)
                    entropies.append(entropy)

                    observations, actions, rewards, values = [], [], [], []
                    if done:  # done and print information
                        max_orn = False
                        if self.max_ep_r < ep_r:
                            self.max_ep_r = ep_r
                            max_orn = True
                        util.ad_process(self.name, self.args, self.lbrain, self.g_ep, self.g_ep_r, ep_r, max_orn, self.res_queue, self.tr_queue, vs, entropies)
                        o = self.env.reset()
                        ep_r = 0.
                        if self.name == 'w00':
                            del vs[:]
                            del entropies[:]
                        break
                o = o_

        self.res_queue.put(None)
        self.tr_queue.put(None)
    
    def test(self, brain, env, args):
        step = 0
        sum_rewards = 0
        while step < args.num_rollout:
            done = False
            o = env.reset()
            while not done:
                if args.render:
                    env.render()
                    time.sleep(0.1)
                a, _, _ = brain.get_action(util.v_wrap(o[None, :]))
                o, r, done, _ = env.step(a)
                sum_rewards += r
                if not args.monitor:
                    if sum_rewards > 195:
                        break
            print('----------------------------------')
            print('total reward of the episode:', sum_rewards)
            print('----------------------------------')
            sum_rewards = 0
            step += 1