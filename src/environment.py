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

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

class Environment(mp.Process):
    def __init__(self, args, gbrain, optimizer, global_ep, global_ep_r, res_queue, tr_queue, name):
        super(Environment, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue, self.tr_queue = global_ep, global_ep_r, res_queue, tr_queue
        self.gbrain, self.optimizer = gbrain, optimizer
        self.env = gym.make('CartPole-v0').unwrapped
        self.lbrain = Policy(self.env.observation_space.shape[0], self.env.action_space.n)           # local network
        self.args = args
        '''
        self.epoch = args.epoch
        self.local_t_max = args.local_t_max
        '''

    def run(self):
        step = 1
        o = self.env.reset()
        ep_r = 0.
        self.max_ep_r = 0.
        entropies = []
        vs = []
        while self.g_ep.value < self.args.epoch:
            observations, actions, values, rewards, probs = [], [], [], [], []
            done = False

            while True:
                '''
                if self.name == 'w00':
                    self.env.render()
                '''
                a, p, v = self.lbrain.get_action(v_wrap(o[None, :]))
                o_, r, done, _ = self.env.step(a)
                if done: r = -1
                ep_r += r
                observations.append(o)
                actions.append(a)
                rewards.append(r)
                values.append(v)
                probs.append(p)


                step += 1

                if step % self.args.local_t_max == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0.               # terminal
                    else:
                        v_s_ = self.lbrain.forward(v_wrap(o_[None, :]))[-1].data.numpy()[0, 0]

                    buffer_v_target = []
                    for r in rewards[::-1]:    # reverse buffer r
                        v_s_ = r + self.args.gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    loss, v_loss, entropy = self.lbrain.loss_func_etp(
                        self.args,
                        v_wrap(np.vstack(observations)),
                        v_wrap(np.array(actions), dtype=np.int64) if actions[0].dtype == np.int64 else v_wrap(np.vstack(actions)),
                        v_wrap(np.array(buffer_v_target)[:, None]))

                    self.optimizer.zero_grad()
                    loss.backward()

                    for lp, gp in zip(self.lbrain.parameters(), self.gbrain.parameters()):
                        gp._grad = lp.grad
                    self.optimizer.step()

                    self.lbrain.load_state_dict(self.gbrain.state_dict())
                    
                    vs.append(v_loss)
                    entropies.append(entropy)

                    observations, actions, rewards = [], [], []
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