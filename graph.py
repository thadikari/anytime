import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import csv
import os


def proc_lines(path, skip_top_n_lines=0):
    return [map(float, line.split(',')) for line in open(path).read().splitlines()[skip_top_n_lines:]]

def proc_csv_obj(path):
    rdr_ = csv.DictReader(open(str(path)))
    cols_ = list([float(row[fn_]) for fn_ in rdr_.fieldnames] for row in rdr_)
    ppk = lambda:0
    ppk.__dict__ = dict(zip(rdr_.fieldnames, zip(*cols_)))
    return ppk

data_dir = 'current'
dir_regex = 'cifar10_*'
# dir_regex = 'mnist_*'

get_files = lambda name: Path('../data/%s'%data_dir).glob('%s/%s.csv'%(dir_regex, name))
gs = gridspec.GridSpec(2, 2)


ax = plt.subplot(gs[:, 0])
for path in reversed(list(get_files('worker_stats'))):
    ll = proc_lines(str(path), 1)
    cols = list(zip(*ll))
    if cols:
        arr = np.array(cols[2])
        n_bins = int((max(arr)-min(arr))/100)
        n, bins, patches = plt.hist(arr, alpha=.5, label=path.parent.name)
    # heights, bins = np.histogram(arr, n_bins)
    # max_height = max(heights)
    # heights = heights/max_height
    # center = (bins[:-1] + bins[1:]) / 2
    # width = bins[1] - bins[0]
    # plt.bar(center, heights, width=width, alpha=.5, label=path.parent.name)
# plt.xlim([0, 3700])
plt.ylabel('Histogram of # of occurrences')
plt.xlabel('Time (ms)')
plt.legend()

for path in get_files('stats'):
    dd = proc_csv_obj(path)
    def plt_(pos_, x_, axis_, lim):
        ax = plt.subplot(gs[pos_, 1])
        # plt.plot(x_, 1-np.array(test_accuracy), label=path.parent.name)
        plt.plot(x_, dd.loss, label=path.parent.name)
        plt.ylabel('Training loss')
        plt.xlabel(axis_)
        # plt.xlim([0, lim])
        # plt.ylim([2, 3.5])
        plt.legend()
    plt_(0, dd.step, 'Step', 40)
    plt_(1, dd.time, 'Time (s)', 400)
    # ax = plt.subplot(gs[1, 1])
    # plt.plot(time, dd.step)

plt.show()

