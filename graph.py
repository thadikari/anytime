from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import scipy.stats as stats
from cycler import cycler
from pathlib import Path
import numpy as np
import argparse
import json
import csv
import os


# print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
# plt.rc('axes', prop_cycle=cycler(color=['r', 'b', 'g', 'y']))
plt.style.use('classic')


def proc_csv(file_path):
    rdr_ = csv.DictReader(open(str(file_path)))
    cols_ = list([float(row[fn_]) for fn_ in rdr_.fieldnames] for row in rdr_)
    # ppk = lambda:0 # ppk.__dict__ =
    return dict(zip(rdr_.fieldnames, zip(*cols_))) if cols_ else {}


_dir_name = '700_mnist_v4'
_dir_regex = 'mnist_*'
# _dir_regex = 'mnist__*'
# 0.63661977236*0.4*0.3 + 0.4 + 0.5

_dir_path = '../data/%s'%_dir_name
_dir_list = list(map(str, Path(_dir_path).glob(_dir_regex)))

def get_label(dir_name):
    if args.short_label:
        if '_fmb_' in dir_name: return 'FMB'
        if '_amb_' in dir_name: return 'AMB'
    else:
        return dir_name


def get_color(dir_name):
    # return None
    if '_fmb_' in dir_name: return 'r'
    if '_amb_' in dir_name: return 'b'


def worker_stats():
    # paths = list(reversed(list(get_paths(dir_regex))))
    proc_worker = lambda dir_path: proc_csv(os.path.join(dir_path, 'worker_stats.csv'))
    proc_args = lambda dir_path: json.load(open(os.path.join(dir_path, 'args')))
    data = list(zip(_dir_list, map(proc_worker, _dir_list), map(proc_args, _dir_list)))

    def hist_(ax_, key, x_label, binwidth=None):
        ylim = 0
        for dir_path, dd, aa in data:
            col = dd.get(key, [])
            if col:
                arr = np.array(col)
                if callable(binwidth): binwidth = binwidth(aa)
                bins = np.arange(min(arr), max(arr) + binwidth, binwidth)
                if (len(bins))==1: bins = [bins[0]-binwidth, bins[0]]

                name = Path(dir_path).name
                n, bins, patches = ax_.hist(arr, bins=bins, alpha=.5, color=get_color(name), label=get_label(Path(dir_path).name))
                if len(bins)>5: ylim = max(ylim, max(n))
        # ax_.set_xlim([10, 18])
        # if args.hist_ylim is not None: ax_.set_ylim([0, args.hist_ylim])
        ax_.set_ylim([0, ylim])
        ax_.set_xlabel(x_label)
        ax_.set_ylabel('Frequency')
        ax_.set_yticks([])
        ax_.legend(loc='best')

    def cum_(ax_):
        x_key, y_key, x_label, y_label = 'step', 'num_samples', 'Step', 'Cumulative sum of examples'
        for dir_path, dd, aa in data:
            mul_ = aa['batch_size']
            name = Path(dir_path).name
            ax_.plot(dd[x_key], np.cumsum(dd[y_key])*mul_, color=get_color(name),
                                                           label=get_label(Path(dir_path).name))
            # print(name, max(dd[x_key]), sum(dd[y_key]), sum(dd[y_key])/max(dd[x_key]))
        ax_.set_xlabel(x_label)
        ax_.set_ylabel(y_label)
        # ax_.set_yticks([])
        ax_.legend(loc='best')
        ax_.grid(True, which='both')

    return {'hist_compute_time': (lambda ax_: hist_(ax_, 'compute_time_worker', 'Computation time (s)', binwidth=0.1)),
            'hist_batch_size': (lambda ax_: hist_(ax_, 'num_samples', 'Batch size',
                                                   binwidth=lambda aa: aa['batch_size']/aa['amb_num_splits']
                                                            # computing split_size 
                                                            )),
            'cumsum_vs_step':(lambda ax_: cum_(ax_)),
            }


def master_stats():
    proc_stats = lambda dir_path: proc_csv(os.path.join(dir_path, 'master_stats.csv'))
    data = list(zip(_dir_list, map(proc_stats, _dir_list)))

    def plot_(ax_, x_key, y_key, x_label, y_label, filter=0):
        for dir_path, dd in data:
            name = Path(dir_path).name
            if filter and args.filter_sigma:
                y_val = gaussian_filter1d(dd[y_key], sigma=args.filter_sigma)
            else:
                y_val = dd[y_key]
            ax_.plot(dd[x_key], y_val, color=get_color(name), label=get_label(name))
        ax_.set_xlabel(x_label)
        ax_.set_ylabel(y_label)
        ax_.legend(loc='best')
        ax_.grid(True, which='both')
        # ax_.set_xscale('log')

        # ax_.xlim([0, lim])
        # ax_.ylim([2, 3.5])
        # ax_.plot(time, dd.step)

    return {'loss_vs_time':(lambda ax_: plot_(ax_, 'time', 'loss', 'Wall clock time (s)', 'Training loss', True)),
            'accuracy_vs_time':(lambda ax_: plot_(ax_, 'time', 'accuracy', 'Wall clock time (s)', 'Test accuracy', True)),
            'loss_vs_step':(lambda ax_: plot_(ax_, 'step', 'loss', 'Step', 'Training loss', True)),
            'time_vs_step':(lambda ax_: plot_(ax_, 'step', 'time', 'Step', 'Wall clock time (s)')),
            'learning_rate_vs_step':(lambda ax_: plot_(ax_, 'step', 'learning_rate', 'Step', 'Learning rate')),
            }


def distribution(ax_):
    proc_dist = lambda dir_path: json.load(open(os.path.join(dir_path, 'args')))['dist']
    data = list(map(proc_dist, _dir_list))
    # if not data: return
    dd = data[0]
    means, std_devs, weights = zip(*dd)
    y_ = lambda x_: sum(w*stats.norm.pdf(x_, mu, sigma) for mu, sigma, w in dd if sigma!=0)
    x = np.linspace(1e-3, 8, 500)
    y = y_(x) + y_(-x) # take absolute values for negative values
    ax_.fill_between(x, 0, y)#, label='Expected value ~=%g'%(x@y/sum(y)))
    ax_.set_xlim([0, max(x)])
    # ax_.ylim([0, 0.0015])
    ax_.set_yticks([])
    ax_.set_ylabel('PDF')
    ax_.set_xlabel('Induced delay (s)')
    # ax_.legend(loc='best')


def all_plots():
    ws, ms = worker_stats(), master_stats()
    all = {**ws, **ms}
    all['distribution'] = distribution
    return all


def main():
    if args.type==1:
        master_stats()[args.plot](plt.gca())
        plot_name = args.plot

    elif args.type==2:
        gs = gridspec.GridSpec(2, 1)
        ms = master_stats()
        ms['loss_vs_step'](plt.subplot(gs[0, 0]))
        ms['loss_vs_time'](plt.subplot(gs[1, 0]))
        plot_name = 'loss'

    elif args.type==0:
        gs = gridspec.GridSpec(3, 3)
        # if args.silent: 
        plt.figure(figsize=(25,15))
        p_ = lambda *k_: [all_plots()[k_[3*i+j]](plt.subplot(gs[j,i]))\
                                        for i in range(3)\
                                            for j in range(3)\
                                                if k_[3*i+j] is not None]

        p_('distribution', 'hist_compute_time', 'hist_batch_size',
           'loss_vs_time', 'accuracy_vs_time', 'cumsum_vs_step',
           'loss_vs_step', 'learning_rate_vs_step', 'time_vs_step')
        plot_name = 'all_plots'

    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()

    if args.save:
        img_path = os.path.join(_dir_path, '%s.pdf'%plot_name)
        plt.savefig(img_path, bbox_inches='tight')
        if args.type==0:
            for key, func in all_plots().items():
                img_path = os.path.join(_dir_path, '%s.pdf'%key)
                plt.figure()
                func(plt.gca())
                plt.savefig(img_path, bbox_inches='tight')

    if not args.silent: plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--short_label', action='store_true')
    parser.add_argument('--plot', default='loss_vs_step',
                        choices=['loss_vs_step', 'loss_vs_time', 'accuracy_vs_time', 'time_vs_step'])
    # parser.add_argument('--hist_ylim', help='y-axis upper limit for histograms', type=float)
    parser.add_argument('--filter_sigma', default=0, type=float)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--silent', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main()
