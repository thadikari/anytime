import collections
import time
import csv
import os


class Registry:
    def __init__(self): self.dd = {}
    def keys(self): return list(self.dd.keys())
    def values(self): return list(self.dd.values())
    def items(self): return self.dd.items()
    def get(self, key): return self.dd[key]
    def put(self, key, val):
        # print(self.keys())
        assert(key not in self.dd)
        self.dd[key] = val
    def reg(self, tp):
        self.put(tp.__name__, tp)
        return tp


def resolve_data_dir(proj_name):
    SCRATCH = os.environ.get('SCRATCH', None)
    if not SCRATCH: SCRATCH = os.path.join(os.path.expanduser('~'), 'SCRATCH')
    return os.path.join(SCRATCH, proj_name)


def resolve_data_dir_os(proj_name, extra=[]):
    if os.name == 'nt': # if windows
        curr_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(curr_path, '..', 'data', *extra)
    else:
        return resolve_data_dir(proj_name)


#https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex
def mpl_init(font_size=14, legend_font_size=None, modify_cycler=True, tick_size=None):
    import matplotlib.pyplot as plt
    from cycler import cycler
    import matplotlib

    custom_cycler = (cycler(color=['r', 'b', 'g', 'y', 'k']) +
                     cycler(linestyle=['-', '--', ':', '-.', '-']))
    if modify_cycler: plt.rc('axes', prop_cycle=custom_cycler)
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams.update({'font.size': font_size})
    if legend_font_size: plt.rc('legend', fontsize=legend_font_size)    # legend fontsize
    # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    if tick_size:
        matplotlib.rcParams['xtick.labelsize'] = tick_size
        matplotlib.rcParams['ytick.labelsize'] = tick_size
    # https://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller

def fmt_ax(ax, xlab, ylab, leg, grid=1):
    if leg: ax.legend(loc='best')
    if xlab: ax.set_xlabel(xlab)
    if ylab: ax.set_ylabel(ylab)
    if grid: ax.grid(alpha=0.7, linestyle='-.', linewidth=0.3)
    ax.tick_params(axis='both')

def save_show_fig(args, plt, file_path):
    plt.tight_layout()
    if args.save:
        for ext in args.ext:
            plt.savefig('%s.%s'%(file_path,ext), bbox_inches='tight')
    if not args.silent: plt.show()

def bind_fig_save_args(parser):
    parser.add_argument('--silent', help='do not show plots', action='store_true')
    parser.add_argument('--save', help='save plots', action='store_true')
    exts_ = ['png', 'pdf']
    parser.add_argument('--ext', help='plot save extention', nargs='*', default=exts_, choices=exts_)


class CSVFile:
    def __init__(self, file_name, work_dir=None, header=None):
        path = file_name if work_dir is None else os.path.join(work_dir, file_name)
        self.fp = open(path, 'a')
        self.csv = csv.writer(self.fp, delimiter=',')
        if header is not None: self.csv.writerow(header)
        self.header = header

    def writerow(self, line, flush=False):
        self.csv.writerow(line)
        if flush: self.flush()

    def flush(self): self.fp.flush()

    def __del__(self):
        self.fp.flush()
        self.fp.close()


class WorkerProfiler:
    def __init__(self, dump_freq, columns, work_dir=None):
        self.reset()
        self.step_count = 0
        self.dump_freq = dump_freq
        self.csv = None if (work_dir is None) else \
                   CSVFile('worker_stats.csv', work_dir, \
                           ['time', 'step'] + columns + ['compute_time_master'])

    def reset(self):
        self.cache = []

    def __enter__(self):
        if self.step_count==0: self.training_started = time.time()
        self.round_started = time.time()
        self.step_count += 1
        return self

    def on_result(self, data):
        curr = time.time()
        elapsed_total = curr - self.training_started
        compute_time_master = curr - self.round_started
        self.cache.append([elapsed_total, self.step_count] + data + [compute_time_master])

    def __exit__(self, type, value, traceback):
        if self.step_count%self.dump_freq==0: self.dump_all()

    def dump_all(self):
        if self.csv is not None:
            for it in self.cache: self.csv.writerow(it)
            self.csv.flush()
            self.reset()

    def __del__(self): self.dump_all()


class Timer:
    def __init__(self): self.updated = None
    def reset(self): self.updated = time.time()
    def elapsed(self): return (time.time() - self.updated)*1000


class LoopProfiler:

    extra_cols = ['TOTAL']

    class Tag:
        def __init__(self, name, line, prof):
            self.name, self.line, self.prof = name, line, prof
            self.timer = Timer()

        def __enter__(self):
            self.timer.reset()
            extr = '' if self.line is None else ': ' + self.line
            self.prof.debg("(( '"+ self.name +"'" + extr)
            return self

        def __exit__(self, type, value, traceback):
            elapsed = self.timer.elapsed()
            self.prof.debg('    elapsed[%s] ))'%str(int(elapsed)))
            self.prof.update(self.name, elapsed)

    def update(self, name, elapsed):
        self.tags[name] = self.tags.get(name, 0) + elapsed

    def __init__(self, print_info, print_debug, csv, dump_freq):
        self.info = print_info
        self.debg = lambda l_: print_debug(l_) if print_debug else None
        self.timer = Timer()
        self.dump_freq = dump_freq
        self.tags = collections.OrderedDict()
        self.step_count = 0
        self.csv = csv

    def __enter__(self):
        self.step_count += 1
        self.timer.reset()
        return self

    def tag(self, name, line=None):
        return LoopProfiler.Tag(name, line, self)

    def __exit__(self, type, value, traceback):
        self.update('TOTAL', self.timer.elapsed())
        if self.step_count%self.dump_freq==0:
            if self.info:
                summ = ', '.join(["'%s':%d"%(key, int(val)) for key, val in self.tags.items()])
                self.info('Summary at[%d] for[%d]: ['%(self.step_count, self.dump_freq) + summ + ']')
            if self.csv:
                keys = self.csv.header if self.csv.header else list(self.tags)
                vals = [self.tags[key] for key in keys]
                self.csv.writerow(vals)
            for key in self.tags: self.tags[key] = 0


class WorkerTag:
    def __init__(self):
        self.tags = collections.OrderedDict()
        self.started = time.time()
        self.curr = ''

    def tag(self, name):
        now = time.time()
        elapsed = now-self.started
        self.tags[self.curr] = elapsed
        self.started = now
        self.curr = name
        return elapsed

    def get(self, name):
        return self.tags.get(name, -1)
