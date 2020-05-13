from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import json, csv, os, argparse
import scipy.stats as stats
from pathlib import Path
import numpy as np
import pandas
import re

import utilities.mpl as utils
import utilities
import utilities.file


plt.style.use('classic')
utils.init(20, legend_font_size=18, tick_size=16)

def proc_csv(file_path):
    if os.stat(file_path).st_size==0: return {}
    ds = pandas.read_csv(file_path, header=0).to_dict('Series')
    return {key:ds[key].to_numpy() for key in ds}

class DataRoot:
    def __init__(self, dir):
        self.dir = dir
        path_ = lambda s_: os.path.join(_a.data_dir, dir, s_)
        self.master_data = proc_csv(path_('master_stats.csv'))
        self.worker_data = proc_csv(path_('worker_stats.csv'))
        self.args = json.load(open(path_('args.json')))

    def get_label(self):
        dir_name = self.dir
        if _a.short_label:
            if '_fmb_' in dir_name: return 'FMB'
            if '_amb_' in dir_name: return 'AMB'
        elif _a.resub:
            for pattern,repl in _a.resub:
                dir_name = re.sub(pattern,repl,dir_name)
            return dir_name
        else:
            return dir_name

    def get_color(self):
        dir_name = self.dir
        if not _a.short_label: return
        if '_fmb_' in dir_name: return 'r'
        if '_amb_' in dir_name: return 'b'



plt_ax = utilities.Registry()

#########################
# plots using worker_data
#########################

def bandwidth_(data, ax_):
    labels = ['send', 'bcast', 'total', 'both']
    cols = ('last_send', 'last_bcast', 'TOTAL')
    def proc_arr(dd):
        send, bcast, total = (dd[key] for key in cols)
        return (send, bcast, total, send + bcast)

    pt_data = [(bb.args['num_workers'], proc_arr(bb.worker_data)) for bb in data]
    pt_data.sort(key=lambda x: x[0])
    numw, numw_lab_arrs = zip(*pt_data)
    lab_numw_arrs = list(zip(*numw_lab_arrs))
    if 0:
        proc_avg = lambda arrs: list(np.mean(arr) for arr in arrs)
        for numw_arrs,lab in zip(lab_numw_arrs,labels):
            ax_.scatter(numw, proc_avg(numw_arrs), label=lab)
        leg = 1
    else:
        proc_avg = lambda arrs: list(np.mean(arr) for arr in arrs)
        numw_arrs = lab_numw_arrs[-1]
        for x_,y_ in zip(numw,numw_arrs): plt.scatter([x_] * len(y_), y_, marker='_')
        leg = 0

    #ax_.fill_between(numw, mn, mx, color='grey', alpha='0.5')
    utils.fmt_ax(ax_, 'Number of workers', 'Average worker to master time', leg=leg)
    ax_.grid(True, which='both')
    ax_.set_xlim([min(numw)-1, max(numw)+1])

def hist_(data, ax_, key, x_label, binwidth=None):
    ylim = 0
    for root in data:
        dd = root.worker_data
        arr = dd.get(key, [])
        if len(arr):
            if callable(binwidth): binwidth = binwidth(root.args)
            bins = np.arange(min(arr), max(arr) + binwidth, binwidth)
            if (len(bins))==1: bins = [bins[0]-binwidth, bins[0]]

            n, bins, patches = ax_.hist(arr, bins=bins, alpha=.5, color=root.get_color(), label=root.get_label())
            #(n!=0).argmax()
            #if len(bins)>5:
            ylim = max(ylim, max(n))
    ax_.set_ylim([0, ylim])
    ax_.set_yticks([])
    utils.fmt_ax(ax_, x_label, 'Frequency', leg=1)

def cum_(data, ax_):
    ax_.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    x_key, y_key, x_label, y_label = 'step', 'num_samples', 'Step', 'Cumulative sum of examples'
    for root in data:
        dd = root.worker_data
        if y_key not in dd: continue
        mul_ = 1  # aa['batch_size']
        y_val = dd[y_key]
        num_ele = int(len(y_val)*1) #_a.fraction)
        y_val = y_val[:num_ele]
        ax_.plot(dd[x_key][:num_ele], np.cumsum(y_val)*mul_, color=root.get_color(),
                               linewidth=1.5, label=root.get_label())
    utils.fmt_ax(ax_, x_label, y_label, leg=1)
    ax_.grid(True, which='both')


@plt_ax.reg
def hist_compute_time(*args):
    return hist_(*args, 'compute_time', 'Computation time (s)', binwidth=_a.binwidth_time)

@plt_ax.reg
def hist_batch_size(*args):
    return hist_(*args, 'num_samples', 'Batch size', binwidth=_a.binwidth_batch)

@plt_ax.reg
def cumsum_vs_step(*args):
    return cum_(*args)


#########################
# plots using master_data
#########################

def plot_(data, ax_, x_key, y_key, x_label, y_label, filter=True, ysci=False):
    if x_key=='time':
        xmax = max(np.max(root.master_data[x_key]) for root in data)
        div,_, x_label = utils.get_best_time_scale(xmax, x_label)
    else:
        ax_.ticklabel_format(style='sci', axis='x', scilimits=(0,3))
        div = 1

    if ysci: ax_.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    for root in data:
        dd = root.master_data
        num_ele = int(len(dd[y_key])*1) #_a.fraction)
        y_val = dd[y_key][:num_ele]
        if filter and _a.filter_sigma:
            y_val = gaussian_filter1d(y_val, sigma=_a.filter_sigma)
        ax_.plot(dd[x_key][:num_ele]/div, y_val, color=root.get_color(),
                                linewidth=1.5, label=root.get_label())
    utils.fmt_ax(ax_, x_label, y_label, leg=1)
    if _a.ylog: ax_.set_yscale('log')
    ax_.grid(True, which='both')

@plt_ax.reg
def loss_vs_time(*args):
    return plot_(*args, 'time', 'loss', 'Wall clock time', 'Training loss')

@plt_ax.reg
def accuracy_vs_time(*args):
    return plot_(*args, 'time', 'accuracy', 'Wall clock time', 'Test accuracy')

@plt_ax.reg
def accuracy_vs_step(*args):
    return plot_(*args, 'step', 'accuracy', 'Step', 'Test accuracy')

@plt_ax.reg
def loss_vs_step(*args):
    return plot_(*args, 'step', 'loss', 'Step', 'Training loss')

@plt_ax.reg
def time_vs_step(*args):
    return plot_(*args, 'time', 'step', 'Wall clock time', 'Step', filter=False, ysci=True)

@plt_ax.reg
def learning_rate_vs_step(*args):
    return plot_(*args, 'step', 'learning_rate', 'Step', 'Learning rate', filter=False, ysci=True)

@plt_ax.reg
def distribution(data, ax_):
    dd = data[0].args['dist']
    means, std_devs, weights = zip(*dd)
    y_ = lambda x_: sum(w*stats.norm.pdf(x_, mu, sigma) for mu, sigma, w in dd if sigma!=0)
    x = np.linspace(1e-3, 8, 500)
    y = y_(x) + y_(-x)          # take absolute values for negative values
    ax_.fill_between(x, 0, y)   # label='Expected value ~=%g'%(x@y/sum(y)))
    ax_.set_xlim([0, max(x)])
    # ax_.ylim([0, 0.0015])
    ax_.set_yticks([])
    utils.fmt_ax(ax_, 'Induced delay (s)', 'PDF', 0)


#########################
# multiple plts on figure
#########################

plt_fig = utilities.Registry()


def single_plot(plt_ax_handle):
    def inner(data, sv_):
        plt_ax_handle(data, plt.gca())
        sv_()
    return inner
for name_,hdl_ in plt_ax.items(): plt_fig.put(name_, single_plot(hdl_))

@plt_fig.reg
def loss(data, sv_):
    gs = gridspec.GridSpec(2, 1)
    loss_vs_step(data, plt.subplot(gs[0, 0]))
    loss_vs_time(data, plt.subplot(gs[1, 0]))
    sv_()

@plt_fig.reg
def master_bandwidth(data, sv_):
    bandwidth_(data, plt.gca())
    sv_()


all_plots_ll = (distribution, hist_compute_time, hist_batch_size,
               loss_vs_time, accuracy_vs_time, cumsum_vs_step,
               loss_vs_step, learning_rate_vs_step, time_vs_step)

@plt_fig.reg
def all_plots(data, sv_):
    if _a.subset is None: hdls = all_plots_ll
    else: hdls = [plt_ax.get(val) for val in _a.subset]
    axes, fig = utils.get_subplot_axes(_a, len(hdls))
    for i, ax in enumerate(axes): hdls[i](data, ax)
    sv_()

@plt_fig.reg
def all_plots_iter(data, sv_):
    all_plots(data, sv_)
    for hdl_ in all_plots_ll:
        plt.figure()
        hdl_(data, plt.gca())
        sv_(hdl_.__name__)


def main():
    dirs = utilities.file.filter_directories(_a, _a.data_dir)
    if not dirs: exit()
    data = [DataRoot(dir) for dir in dirs]
    get_path = lambda name_: os.path.join(_a.data_dir, name_)
    def save_hdl(name_=_a.type):
        plt.gcf().canvas.set_window_title(f'{_a.data_dir}: {name_}')
        utils.save_show_fig(_a, plt, get_path(name_))
    plt_fig.get(_a.type)(data, save_hdl)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=utilities.file.resolve_data_dir_os('distributed'))

    parser.add_argument('--type', default='all_plots', choices=plt_fig.keys())
    parser.add_argument('--subset', nargs='+', choices=plt_ax.keys())

    parser.add_argument('--binwidth_time', type=float, default=0.01)
    parser.add_argument('--binwidth_batch', type=float, default=1)
    parser.add_argument('--short_label', action='store_true')
    parser.add_argument('--resub', action='append', nargs=2, metavar=('pattern','substitute'))

    parser.add_argument('--ylog', action='store_true')
    parser.add_argument('--filter_sigma', default=0, type=float)
    # parser.add_argument('--fraction', help='drop time series data after this fraction', default=1, type=float)

    utilities.file.bind_dir_filter_args(parser)
    utils.bind_subplot_args(parser, ax_size_default=[8,5])
    utils.bind_fig_save_args(parser)
    return parser.parse_args()

if __name__ == '__main__':
    _a = parse_args()
    print('[Arguments]', vars(_a))
    main()
