import os
import numpy as np
import time
import argparse
import gym
from gym import wrappers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.multiprocessing as mp

from util import v_wrap

def test(brain, args, env):
    step = 0
    sum_rewards = 0
    while step < args.num_rollout:
        done = False
        o = env.reset()
        while not done:
            if args.render:
                env.render()
                time.sleep(0.1)
            a, _, _ = brain.get_action(v_wrap(o[None, :]))
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