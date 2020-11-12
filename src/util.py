from enum import Enum

from tabulate import tabulate
from contextlib import contextmanager
import numpy as np
import os
import os.path as osp
import sys
import datetime
import dateutil.tz
import csv
import joblib
import json
import pickle
import base64

from gym import wrappers
import pytz
import matplotlib.pyplot as plt

csvFILE_NAME = 'progress.csv'
paramFILE_NAME = 'params.json'
plotFILE_NAME = 'a3c_cartpole-v1.png'

today_dirname = ''
started_dirname = ''
nwparam_dirname = ''
mon_dirname = ''

today = ''
started_time = ''

import logger

def ad_process(name, args, brain, g_ep, g_ep_r, ep_r, max_orn, res_queue, tr_queue, vs, entropies):
    with g_ep.get_lock():
        g_ep.value += 1
    with g_ep_r.get_lock():
        if g_ep_r.value == 0.:
            g_ep_r.value = ep_r
        else:
            g_ep_r.value = g_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(g_ep_r.value)
    tr_queue.put(ep_r)
    '''
    print(
        name,
        "Ep:", g_ep.value,
        "| Ep_r: %.0f" % g_ep_r.value,
    )
    '''
    if name == 'w00':
        print('----------------------------------')
        print('name:', name)
        print('step:', g_ep.value)
        print('total reward of the episode:', ep_r)
        print('moving average of reward:', g_ep_r.value)
        print('----------------------------------')
        logger.record_tabular_misc_stat('Entropy', entropies)
        logger.record_tabular_misc_stat('V', vs)
        logger.record_tabular('reward', ep_r)
        logger.record_tabular('step', g_ep.value)
        logger.dump_tabular()
    
    #logger.save_nw_param(args, brain, g_ep, max_orn)
    logger.save_nw_param(args, brain, ep_r, max_orn)
    

def date_to_date(datetime_obj):
    """
    datetime型もしくはdate型をうけとり所定の日付文字列で返す関数
    """
    date_str = datetime_obj.strftime('%Y-%m-%d-%H%M%S')
    # date_str = datetime_obj.strftime('%Y-%m-%d %H/%M/%S')
    return  date_str

def date_to_str(datetime_obj):
    """
    datetime型もしくはdate型をうけとりDateのみの日付文字列で返す関数
    """
    date_str = datetime_obj.strftime('%Y-%m-%d')
    return  date_str

def date_calc(date_str, delta_date):
    """
    日付計算の便利関数
    :param date_str: 所定の日付形式
    :param delta_date: 計算したい日付　範囲は整数
    :return: 計算後の所定の日付形式
    """
    target_date = dateutil.parser.parse(date_str)
    date_temp = target_date + datetime.timedelta(days=delta_date)
    date_str_changed = date_to_str(date_temp)
    return date_str_changed

'''
# 秒数をそのまま返す
def get_japantime_now():
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    return now
'''
# 秒の.(コンマ)以下を削除
def get_japantime_now():
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    return date_to_date(now)

def get_japantime_today():
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    today = now.strftime('%Y-%m-%d')
    return today

def init_mkdir(args):
    global today_dirname, started_dirname, nwparam_dirname, today, started_time, mon_dirname
    today = get_japantime_today()
    started_time = get_japantime_now()
    today_dirname = os.path.join(args.log_dir, today)
    started_dirname = os.path.join(today_dirname, started_time)

    dir_name = started_dirname
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    nwparam_dirname = os.path.join(dir_name, 'network_param')
    if not os.path.exists(nwparam_dirname):
        os.makedirs(nwparam_dirname)

    if args.monitor:
        mon_dirname = os.path.join(dir_name, 'movie')
        if not os.path.exists(mon_dirname):
            os.makedirs(mon_dirname)
        
        
    

def init_setting(env, args):
    init_mkdir(args)
    dir_name = started_dirname
    logger.add_tabular_output(dir_name, csvFILE_NAME)
    logger.log_parameters_lite(dir_name, paramFILE_NAME, args)

    if args.monitor:
        env = wrappers.Monitor(env, mon_dirname, force=True)
    return env

def plot_graph(res, tr_res):
    dir_name = started_dirname

    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

    axL.plot(res)
    axL.set_ylabel('Moving average ep reward')
    axL.set_xlabel('Step')

    axR.plot(tr_res)
    axR.set_ylabel('total reward')
    axR.set_xlabel('Step')

    fig.savefig(os.path.join(dir_name, plotFILE_NAME))