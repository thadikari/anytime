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

    def writerow(self, line, flush=False):
        self.csv.writerow(line)
        if flush: self.flush()

    def flush(self): self.fp.flush()

    def __del__(self):
        self.fp.flush()
        self.fp.close()


class WorkerProfiler:
    def __init__(self, dump_freq, work_dir=None, log=None):
        self.log = log
        self.dump_freq = dump_freq
        self.csv = None if (work_dir is None) else \
                   CSVFile('worker_stats.csv', work_dir, \
                   ['rank', 'elapsed_master (ms)', 'elapsed_worker (ms)', 'difference (ms)'])
        self.reset()

    def reset(self):
        self.step_count = 0
        self.cache = []

    def __enter__(self):
        self.started = time.time()
        self.step_count += 1
        return self

    def on_result(self, rank, elapsed_worker):
        elapsed_master = (time.time() - self.started)*1000
        self.cache.append([rank, elapsed_master, elapsed_worker, elapsed_master-elapsed_worker])

    def __exit__(self, type, value, traceback):
        if self.step_count%self.dump_freq==0 and self.csv is not None:
            for it in self.cache: self.csv.writerow(it)
            self.csv.flush()
            self.reset()


class LoopProfiler:

    class Tag:
        def __init__(self, name, line, prof):
            self.name, self.line, self.prof = name, line, prof

        def elapsed(self):
            return (time.time() - self.updated)*1000

        def __enter__(self):
            self.updated = time.time()
            extr = '' if self.line is None else ': ' + self.line
            self.prof.log.debug("(( '"+ self.name +"'" + extr)
            return self

        def __exit__(self, type, value, traceback):
            elapsed = self.elapsed()
            self.prof.log.debug('    elapsed[%s] ))'%str(int(elapsed)))
            self.prof.tags[self.name] = self.prof.tags.get(self.name, 0) + elapsed

    def __init__(self, log, dump_freq):
        self.log = log
        self.updated = time.time()
        self.dump_freq = dump_freq
        self.tags = collections.OrderedDict()
        self.step_count = 0

    def __enter__(self):
        self.step_count += 1
        return self

    def tag(self, name, line=None):
        return LoopProfiler.Tag(name, line, self)

    def __exit__(self, type, value, traceback):
        if self.step_count%self.dump_freq==0:
            summ = ', '.join(["'%s':%d"%(key, int(val)) for key, val in self.tags.items()])
            self.log.info('Summary at[%d] for[%d]: ['%(self.step_count, self.dump_freq) + summ + ']')
            for key in self.tags: self.tags[key] = 0
