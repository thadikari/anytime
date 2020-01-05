import collections
import time
import csv
import os


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
                vals = [self.tags[key] for key in keys + ['TOTAL']]
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
