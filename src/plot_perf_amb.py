from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import json, csv, os, argparse
import scipy.stats as stats
from pathlib import Path
import numpy as np
import re

import utilities.mpl as utils
import utilities
import utilities.file


#from matplotlib import ticker
#ax_.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: '{:,g}'.format(x/10**6) + 'M'))
plt.style.use('classic')
utils.init(20, legend_font_size=18, tick_size=16)

def proc_csv(file_path):
    rdr_ = csv.DictReader(open(str(file_path)))
    cols_ = list([float(row[fn_]) for fn_ in rdr_.fieldnames] for row in rdr_)
    return dict(zip(rdr_.fieldnames, zip(*cols_))) if cols_ else {}

proc_master = lambda dir_path: proc_csv(os.path.join(dir_path, 'master_stats.csv'))
proc_worker = lambda dir_path: proc_csv(os.path.join(dir_path, 'worker_stats.csv'))
proc_args = lambda dir_path: json.load(open(os.path.join(dir_path, 'args.json')))
proc_dist = lambda dir_path: proc_args(dir_path)['dist']

class DataRoot:
    def __init__(self, dir_list):
        self.__wd, self.__md = None, None
        self.dir_list = dir_list

    def worker_data(self):
        if self.__wd is None:
            self.__wd = list(zip(self.dir_list, map(proc_worker, self.dir_list),
                                   map(proc_args, self.dir_list)))
        return self.__wd

    def master_data(self):
        if self.__md is None:
            self.__md = list(zip(self.dir_list, map(proc_master, self.dir_list)))
        return self.__md

    def distribution(self):
        return list(map(proc_dist, self.dir_list))

def get_label(dir_name):
    if _a.short_label:
        if '_fmb_' in dir_name: return 'FMB'
        if '_amb_' in dir_name: return 'AMB'
    elif _a.resub:
        for pattern,repl in _a.resub:
            dir_name = re.sub(pattern,repl,dir_name)
        return dir_name
    else:
        return dir_name

def get_color(dir_name):
    if not _a.short_label: return
    if '_fmb_' in dir_name: return 'r'
    if '_amb_' in dir_name: return 'b'



plt_ax = utilities.Registry()

#########################
# plots using worker_data
#########################

def bandwidth_(root, ax_):
    labels = ['send', 'bcast', 'total', 'both']
    cols = ('last_send', 'last_bcast', 'TOTAL')
    def proc_arr(dd):
        send, bcast, total = (np.array(dd[key]) for key in cols)
        return (send, bcast, total, send + bcast)

    pt_data = [(aa['num_workers'], proc_arr(dd)) for _, dd, aa in root.worker_data()]
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

def hist_(root, ax_, key, x_label, binwidth=None):
    ylim = 0
    for dir_path, dd, aa in root.worker_data():
        col = dd.get(key, [])
        if col:
            arr = np.array(col)
            if callable(binwidth): binwidth = binwidth(aa)
            bins = np.arange(min(arr), max(arr) + binwidth, binwidth)
            if (len(bins))==1: bins = [bins[0]-binwidth, bins[0]]

            name = Path(dir_path).name
            n, bins, patches = ax_.hist(arr, bins=bins, alpha=.5, color=get_color(name), label=get_label(Path(dir_path).name))
            if len(bins)>5: ylim = max(ylim, max(n))
    ax_.set_ylim([0, ylim])
    ax_.set_yticks([])
    utils.fmt_ax(ax_, x_label, 'Frequency', leg=1)

def cum_(root, ax_):
    ax_.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    x_key, y_key, x_label, y_label = 'step', 'num_samples', 'Step', 'Cumulative sum of examples'
    for dir_path, dd, aa in root.worker_data():
        if y_key not in dd: continue
        mul_ = 1  # aa['batch_size']
        name = Path(dir_path).name
        y_val = dd[y_key]
        num_ele = int(len(y_val)*_a.fraction)
        y_val = y_val[:num_ele]
        ax_.plot(dd[x_key][:num_ele], np.cumsum(y_val)*mul_, color=get_color(name),
                               linewidth=1.5, label=get_label(Path(dir_path).name))
    utils.fmt_ax(ax_, x_label, y_label, leg=1)
    ax_.grid(True, which='both')


@plt_ax.reg
def hist_compute_time(*args):
    return hist_(*args, 'compute_time', 'Computation time (s)', binwidth=0.1)

@plt_ax.reg
def hist_batch_size(*args):
    return hist_(*args, 'num_samples', 'Batch size',
             binwidth=lambda aa: aa['batch_size']/aa['amb_num_partitions'])

@plt_ax.reg
def cumsum_vs_step(*args):
    return cum_(*args)


#########################
# plots using master_data
#########################

def plot_(root, ax_, x_key, y_key, x_label, y_label, filter=0, ysci=False):
    if ysci: ax_.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    for dir_path, dd in root.master_data():
        name = Path(dir_path).name
        num_ele = int(len(dd[y_key])*_a.fraction)
        y_val = dd[y_key][:num_ele]
        if filter and _a.filter_sigma:
            y_val = gaussian_filter1d(y_val, sigma=_a.filter_sigma)
        ax_.plot(dd[x_key][:num_ele], y_val, color=get_color(name), 
                                linewidth=1.5, label=get_label(name))
    utils.fmt_ax(ax_, x_label, y_label, leg=1)
    if _a.ylog: ax_.set_yscale('log')
    ax_.grid(True, which='both')

@plt_ax.reg
def loss_vs_time(*args):
    return plot_(*args, 'time', 'loss', 'Wall clock time (s)', 'Training loss', True)

@plt_ax.reg
def accuracy_vs_time(*args):
    return plot_(*args, 'time', 'accuracy', 'Wall clock time (s)', 'Test accuracy', True)

@plt_ax.reg
def accuracy_vs_step(*args):
    return plot_(*args, 'step', 'accuracy', 'Step', 'Test accuracy', True)

@plt_ax.reg
def loss_vs_step(*args):
    return plot_(*args, 'step', 'loss', 'Step', 'Training loss', True)

@plt_ax.reg
def time_vs_step(*args):
    return plot_(*args, 'step', 'time', 'Step', 'Wall clock time (s)', ysci=True)

@plt_ax.reg
def learning_rate_vs_step(*args):
    return plot_(*args, 'step', 'learning_rate', 'Step', 'Learning rate', ysci=True)

@plt_ax.reg
def distribution(root, ax_):
    dd = root.distribution()[0]
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
    def inner(root, sv_):
        plt_ax_handle(root, plt.gca())
        sv_()
    return inner
for name_,hdl_ in plt_ax.items(): plt_fig.put(name_, single_plot(hdl_))

@plt_fig.reg
def loss(root, sv_):
    gs = gridspec.GridSpec(2, 1)
    loss_vs_step(root, plt.subplot(gs[0, 0]))
    loss_vs_time(root, plt.subplot(gs[1, 0]))
    sv_()

@plt_fig.reg
def master_bandwidth(root, sv_):
    bandwidth_(root, plt.gca())
    sv_()


all_plots_ll = (distribution, hist_compute_time, hist_batch_size,
               loss_vs_time, accuracy_vs_time, cumsum_vs_step,
               loss_vs_step, learning_rate_vs_step, time_vs_step)

@plt_fig.reg
def all_plots(root, sv_):
    if _a.subset is None: hdls = all_plots_ll
    else: hdls = [plt_ax.get(val) for val in _a.subset]
    axes, fig = utils.get_subplot_axes(_a, len(hdls))
    for i, ax in enumerate(axes): hdls[i](root, ax)
    sv_()

@plt_fig.reg
def all_plots_iter(root, sv_):
    all_plots(root, sv_)
    for hdl_ in all_plots_ll:
        plt.figure()
        hdl_(root, plt.gca())
        sv_(hdl_.__name__)


def main():
    root = DataRoot(_a.dir_list)
    get_path = lambda name_: os.path.join(_a.dir_path, name_)
    def save_hdl(name_=_a.type):
        plt.gcf().canvas.set_window_title(f'{_a.dir_path}: {name_}')
        utils.save_show_fig(_a, plt, get_path(name_))
    plt_fig.get(_a.type)(root, save_hdl)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=utilities.file.resolve_data_dir_os('distributed'))
    parser.add_argument('--dir_name', default='', type=str)
    parser.add_argument('--dir_regex', default='*', type=str)

    parser.add_argument('--type', default='all_plots', choices=plt_fig.keys())
    parser.add_argument('--subset', nargs='+', choices=plt_ax.keys())

    parser.add_argument('--short_label', action='store_true')
    parser.add_argument('--resub', action='append', nargs=2, metavar=('patter','substitute'))

    parser.add_argument('--ylog', action='store_true')
    parser.add_argument('--filter_sigma', default=0, type=float)
    parser.add_argument('--fraction', help='drop time series data after this fraction', default=1, type=float)

    utils.bind_subplot_args(parser, ax_size_default=[8,5])
    utils.bind_fig_save_args(parser)
    return parser.parse_args()

if __name__ == '__main__':
    _a = parse_args()
    _a.dir_path = os.path.join(_a.data_dir, _a.dir_name)
    _a.dir_list = list(str(p_) for p_ in Path(_a.dir_path).glob(_a.dir_regex) if p_.is_dir())
    print('[Arguments]', vars(_a))
    if not len(_a.dir_list):
        print(f'Empty data directory!!!: {_a.dir_path}.')
        print('Use --data_dir and --dir_name (e.g. 800_cifar10/set2) to point to a directory with data.')
        print('Filter directories using --dir_regex (e.g. cifar10_*).')
    else: main()
