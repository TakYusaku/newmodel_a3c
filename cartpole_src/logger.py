from enum import Enum
import os
import os.path as osp
import sys
import torch

from tabulate import tabulate
from contextlib import contextmanager
import numpy as np
import datetime
import dateutil.tz
import csv
import joblib
import json
import pickle
import base64

import util

_prefixes = []
_prefix_str = ''

_tabular_prefixes = []
_tabular_prefix_str = ''

_tabular = []

_text_outputs = []
_tabular_outputs = []

_text_fds = {}
_tabular_fds = {}
_tabular_header_written = set()

_snapshot_dir = None
_snapshot_mode = 'all'
_snapshot_gap = 1

_log_tabular_only = False
_header_printed = False


def _add_output(dir_name, file_name, arr, fds, mode='a'):
    fn = os.path.join(dir_name, file_name)
    if fn not in arr:
        arr.append(fn)
        fds[fn] = open(fn, mode)

def add_tabular_output(dir_name, file_name):
    _add_output(dir_name, file_name, _tabular_outputs, _tabular_fds, mode='w')

class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {'$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name}
        return json.JSONEncoder.default(self, o)

def log(s, with_prefix=True, with_timestamp=True):#, color=None):
    out = s
    if with_prefix:
        out = _prefix_str + out
    if with_timestamp:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
        out = "%s | %s" % (timestamp, out)
    '''
    if color is not None:
        out = colorize(out, color)
    '''
    if not _log_tabular_only:
        # Also log to stdout
        # print(out)
        for fd in list(_text_fds.values()):
            fd.write(out + '\n')
            fd.flush()
        sys.stdout.flush()

def log_parameters_lite(dir_name, log_file, args):
    log_params = {}
    fn = os.path.join(dir_name, log_file)
    log_params['started_time'] = util.started_time
    for param_name, param_value in args.__dict__.items():
        if param_name == 'log_dir':
            param_value = util.started_dirname
        log_params[param_name] = param_value
    with open(fn, "w") as f:
        json.dump(log_params, f, indent=2, sort_keys=True, cls=MyEncoder)

def pop_prefix():
    del _prefixes[-1]
    global _prefix_str
    _prefix_str = ''.join(_prefixes)

def push_prefix(prefix):
    _prefixes.append(prefix)
    global _prefix_str
    _prefix_str = ''.join(_prefixes)

def push_tabular_prefix(key):
    _tabular_prefixes.append(key)
    global _tabular_prefix_str
    _tabular_prefix_str = ''.join(_tabular_prefixes)


def pop_tabular_prefix():
    del _tabular_prefixes[-1]
    global _tabular_prefix_str
    _tabular_prefix_str = ''.join(_tabular_prefixes)

@contextmanager
def prefix(key):
    push_prefix(key)
    try:
        yield
    finally:
        pop_prefix()

@contextmanager
def tabular_prefix(key):
    push_tabular_prefix(key)
    yield
    pop_tabular_prefix()

def record_tabular(key, val):
    _tabular.append((_tabular_prefix_str + str(key), str(val)))

def record_tabular_misc_stat(key, values):
    record_tabular(key + "Average", np.average(values))
    record_tabular(key + "Std", np.std(values))
    record_tabular(key + "Median", np.median(values))
    record_tabular(key + "Min", np.amin(values))
    record_tabular(key + "Max", np.amax(values))
'''
def save_nw_param(args, brain, g_ep, max_orn):
    if args.save_mode == 'all':
        torch.save(brain, os.path.join(util.nwparam_dirname, args.save_name+"_{}.pkl".format(g_ep.value)))
    elif args.save_mode == 'last':
        torch.save(brain, os.path.join(util.nwparam_dirname, args.save_name+'_last.pkl'))
    elif args.save_mode == 'max':
        if max_orn:
            torch.save(brain, os.path.join(util.nwparam_dirname, args.save_name+'_max_{}.pkl').format(g_ep.value)))
'''
def save_nw_param(args, brain, ep_r, max_orn):
    if args.save_mode == 'all':
        torch.save(brain, os.path.join(util.nwparam_dirname, args.save_name+"_{}.pkl".format(int(ep_r))))
    elif args.save_mode == 'last':
        torch.save(brain, os.path.join(util.nwparam_dirname, args.save_name+'_last.pkl'))
    elif args.save_mode == 'max':
        if max_orn:
            torch.save(brain, os.path.join(util.nwparam_dirname, args.save_name+'_max_{}.pkl').format(int(ep_r)))

class TerminalTablePrinter(object):
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


table_printer = TerminalTablePrinter()


def dump_tabular(*args, **kwargs):
    wh = kwargs.pop("write_header", None)
    if len(_tabular) > 0:
        if _log_tabular_only:
            table_printer.print_tabular(_tabular)
        else:
            for line in tabulate(_tabular).split('\n'):
                log(line, *args, **kwargs)
        tabular_dict = dict(_tabular)
        # Also write to the csv files
        # This assumes that the keys in each iteration won't change!
        for tabular_fd in list(_tabular_fds.values()):
            writer = csv.DictWriter(tabular_fd, fieldnames=list(tabular_dict.keys()))
            if wh or (wh is None and tabular_fd not in _tabular_header_written):
                writer.writeheader()
                _tabular_header_written.add(tabular_fd)
            writer.writerow(tabular_dict)
            tabular_fd.flush()
        del _tabular[:]