import time
import csv
import os


class CSVFile:
    def __init__(self, file_name, work_dir=None, header=None):
        path = file_name if work_dir is None else os.path.join(work_dir, file_name)
        self.fp = open(path, 'a')
        self.csv = csv.writer(self.fp, delimiter=',')
        if header is not None: self.csv.writerow(header)

    def writerow(self, line, flush=False):
        self.csv.writerow(line)
        if flush: self.flush()

    def flush(self): self.fp.flush()

    def __del__(self):
        self.fp.flush()
        self.fp.close()


class WorkerProfiler:
    def __init__(self, dump_freq, columns, work_dir=None):
        self.curr = ''
        self.tags = {}
        self.step_count = 0
        self.dump_freq = dump_freq
        self.columns = columns
        self.csv = None if (work_dir is None) else \
                   CSVFile('worker_stats.csv', work_dir, \
                           ['time', 'step'] + columns)

    def reset_begin(self, name):
        self.step_count += 1
        self.started = time.time()
        return self.begin(name)

    def begin(self, name):
        now = time.time()
        elapsed = now-self.started
        self.tags[self.curr] = elapsed
        self.started = now
        self.curr = name
        return elapsed

    def tag(self, name, val):
        self.tags[name] = val

    def end(self):
        self.begin('_')
        stats = [time.time(), self.step_count] +\
                [self.tags[col] for col in self.columns]
        return stats

    def dump(self, stats):
        if self.csv is not None:
            for it in stats:
                self.csv.writerow(it)
            if self.step_count%self.dump_freq==0:
                self.csv.flush()
