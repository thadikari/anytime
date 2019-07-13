import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import argparse
import csv
import os


def proc_lines(path, skip_top_n_lines=0):
    return [map(float, line.split(',')) for line in open(path).read().splitlines()[skip_top_n_lines:]]
    # usage: proc_lines(path, 1)

def proc_csv_obj(path):
    rdr_ = csv.DictReader(open(str(path)))
    cols_ = list([float(row[fn_]) for fn_ in rdr_.fieldnames] for row in rdr_)
    # ppk = lambda:0
    # ppk.__dict__ =
    return dict(zip(rdr_.fieldnames, zip(*cols_))) if cols_ else {}


data_dir = 'current'
dir_regex = 'cifar10_*'
# dir_regex = 'mnist_*'

get_paths = lambda name: map(str, Path('../data/%s'%data_dir).\
                         glob('%s/%s.csv'%(dir_regex, name)))

def plot_hist():
    paths = list(reversed(list(get_paths('worker_stats'))))
    data = list(zip(paths, map(proc_csv_obj, paths)))

    def plot_(ax_, key):
        for path, dd in data:
            col = dd.get(key, [])
            if col:
                arr = np.array(col)
                binwidth = 0.1
                bins = np.arange(min(arr), max(arr) + binwidth, binwidth)
                n, bins, patches = ax_.hist(arr, bins=bins, alpha=.5, label=Path(path).parent.name)
            # heights, bins = np.histogram(arr, n_bins)
            # max_height = max(heights)
            # heights = heights/max_height
            # center = (bins[:-1] + bins[1:]) / 2
            # width = bins[1] - bins[0]
            # plt.bar(center, heights, width=width, alpha=.5, label=path.parent.name)
        ax_.set_xlim([10, 18])
        ax_.set_xlabel('Time (s)')
        ax_.set_ylabel('Histogram of occurrences')
        ax_.legend(loc='best')

    return (lambda ax_: plot_(ax_, 'compute_time_worker'))


def series_plotters():
    paths = list(get_paths('stats'))
    data = list(zip(paths, map(proc_csv_obj, paths)))

    def plot_(ax_, x_key, y_key, x_label, y_label):
        for path, dd in data:
            # ax_.plot(x_, 1-np.array(test_accuracy), label=path.parent.name)
            ax_.plot(dd[x_key], dd[y_key], label=Path(path).parent.name)
        ax_.set_xlabel(x_label)
        ax_.set_ylabel(y_label)
        ax_.legend(loc='best')
        # ax_.xlim([0, lim])
        # ax_.ylim([2, 3.5])
        # ax_.plot(time, dd.step)

    return {'loss_step':(lambda ax_: plot_(ax_, 'step', 'loss', 'Step', 'Training loss')),
            'loss_time':(lambda ax_: plot_(ax_, 'time', 'loss', 'Time (s)', 'Training loss')),
            'step_time':(lambda ax_: plot_(ax_, 'time', 'step', 'Time (s)', 'Step')),
            'accuracy_time':(lambda ax_: plot_(ax_, 'time', 'accuracy', 'Time (s)', 'Test accuracy')),
            }

def main(args):
    if args.grid==1:
        series_plotters()[args.plot](plt.gca())

    elif args.grid==2:
        gs = gridspec.GridSpec(2, 1)
        sp = series_plotters()
        sp['loss_step'](plt.subplot(gs[0, 0]))
        sp['loss_time'](plt.subplot(gs[1, 0]))

    elif args.grid==3:
        gs = gridspec.GridSpec(2, 2)
        plot_hist()(plt.subplot(gs[:, 0]))
        sp = series_plotters()
        sp['loss_step'](plt.subplot(gs[0, 1]))
        sp['loss_time'](plt.subplot(gs[1, 1]))

    elif args.grid==4:
        gs = gridspec.GridSpec(2, 2)
        plot_hist()(plt.subplot(gs[0, 0]))
        sp = series_plotters()
        sp['step_time'](plt.subplot(gs[1, 0]))
        sp['loss_step'](plt.subplot(gs[0, 1]))
        sp['loss_time'](plt.subplot(gs[1, 1]))

    elif args.grid==5:
        gs = gridspec.GridSpec(2, 2)
        plot_hist()(plt.subplot(gs[0, 0]))
        sp = series_plotters()
        # sp['step_time'](plt.subplot(gs[1, 0]))
        sp['loss_step'](plt.subplot(gs[0, 1]))
        sp['accuracy_time'](plt.subplot(gs[1, 0]))
        sp['loss_time'](plt.subplot(gs[1, 1]))

    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('grid', type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--plot', default='loss_step', choices=['loss_step', 'loss_time', 'accuracy_time', 'step_time'])
    return parser.parse_args()

if __name__ == "__main__": main(parse_args())
