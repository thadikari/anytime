import numpy as np
import logging
import pickle
import socket
import time

import utilities as ut


MPI = None
comm = None
log = None
ROOT_RANK = 0

def init(log_level=logging.INFO): # INFO DEBUG

    global MPI, comm, log

    logger = ut.misc.setup_logger(logging.getLogger('hvd'))
    hostname = socket.gethostname()
    host = socket.gethostbyname(hostname)
    logger.info('Initializing...')

    from mpi4py import MPI as MPI_
    MPI = MPI_
    comm = MPI.COMM_WORLD

    log = logger.getChild('wk%d'%rank())
    log.info('Initialized rank [%d], hostname [%s], host [%s]', rank(), str(hostname), str(host))


def is_master(): return rank()==ROOT_RANK
def rank(): return comm.Get_rank()
def size(): return comm.Get_size()
def num_workers(): return size()-1


# mpi4py references
# [1] https://github.com/thadikari/scripts/blob/master/tests/mpi4py/Ibcast_irecv.py
# [2] https://github.com/mpi4py/mpi4py/blob/master/src/mpi4py/MPI/Comm.pyx
# [3] https://stackoverflow.com/questions/10882581/mpi-isend-request-parameter
# [4] https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/
# [5] https://nyu-cds.github.io/python-mpi/05-collectives/
# [6] https://nyu-cds.github.io/python-mpi/05-collectives/




###############################################################################

class Synchronous:
    def __init__(self, work_dir): self.work_dir = work_dir
    def make_master(self, csv_cols): return SynchronousMaster(csv_cols, self.work_dir)
    def make_worker(self): return SynchronousWorker()


def default_bcast_func(*data):
    log.debug('Broadcast from [%d], rank [%d]', ROOT_RANK, rank())
    return comm.bcast(data, root=ROOT_RANK)

def mpi_reduce_grads(grads, num_samples, meta):
    meta_ll = comm.gather(meta, root=ROOT_RANK) # reference [4,5]
    total = comm.reduce(num_samples, op=MPI.SUM, root=ROOT_RANK)
    for acc in grads: # reference [6]
        comm.Reduce(acc*(0. if rank()==0 else 1.), acc, op=MPI.SUM, root=ROOT_RANK)
    return total, meta_ll


class MasterBase:
    def __init__(self, csv_cols, work_dir):
        self.work = WorkerProfiler(5, csv_cols, work_dir)

class SynchronousMaster(MasterBase):
    def collect_grads(self, grads):
        log.debug('Listening to [%d] workers', num_workers())
        total, meta_ll = mpi_reduce_grads(grads, 0., None)
        for acc in grads: np.divide(acc, total, out=acc)
        with self.work as ww:
            # mpi gather always ensures the rank order?? Yes as per [4]
            for i in range(num_workers()): ww.on_result(meta_ll[i+1])
            # i+1 to skip master

    def broadcast_vars(self, *vars): return default_bcast_func(*vars)


class SynchronousWorker:
    def dispatch_grads(self, *args): return mpi_reduce_grads(*args)
    def update_vars(self, *vars): return default_bcast_func(*vars)




###############################################################################

class Asynchronous:
    def __init__(self, master_style, master_batch_min, master_time_limit, work_dir):
        def make_master(csv_cols):
            if master_style=='batch': ch = AsyncMasterBatch(master_batch_min)
            elif master_style=='time': ch = AsyncMasterTime(master_time_limit)
            return AsynchronousMaster(csv_cols, work_dir).init(ch)

        self.make_master = make_master

    def make_worker(self): return AsynchronousWorker()


class ReqMan:
    def __init__(self, name='ReqMan', max_size=-1):
        self.name, self.reqs = name, []
        self.max_size = max_size
        self.log = log.getChild(self.name)

    def add(self, req): self.addn([req])

    def addn(self, reqs): # list of requests
        self.reqs.extend(reqs)
        self.remove_completed()
        l_ = lambda: len(self.reqs)
        self.log.info('Queue size [%d]', l_())

        if self.max_size>0 and l_()>self.max_size:
            self.log.info('Throttling [%d]', l_())
            while l_() > max(self.max_size*0.75, 1):
                time.sleep(0.001)  # 1 millisecond
                self.remove_completed()
            self.log.info('Throttle end [%d]', l_())

    def remove_completed(self):
        # test() returns (flag, msg) whereas Test() returns flag only. See reference [2].
        # if Test() is true the resources are automatically freed as per [3].
        # instead of storing requests, can also just free them as mentioned in [3].
        self.reqs = [req for req in self.reqs if not req.Test()]


class ReqManMax1:
    def __init__(self, name=None): self.name, self.last_req = name, None
    def add(self, req):
        if not self.last_req is None: self.last_req.Wait()
        self.last_req = req


class AsyncMasterBatch:
    def __init__(self, threshold): self.threshold = threshold
    def reset(self): self.count = 0
    def should_wait(self):
        self.count += 1
        return self.count <= self.threshold


class AsyncMasterTime:
    def __init__(self, threshold): self.threshold = threshold
    def reset(self): self.time = time.time()
    def should_wait(self): return time.time()-self.time <= self.threshold


class AsynchronousMaster(MasterBase):
    def init(self, checker):
        self.reqman = ReqMan('BCAST')
        # self.buff = bytearray(1<<30)
        self.checker = checker
        return self

    def collect_grads(self, grads):
        total = 0
        self.checker.reset()
        while self.checker.should_wait():
            state = MPI.Status()
            ngrads, num_samples, meta = comm.recv(source=MPI.ANY_SOURCE, status=state)
            # ngrads, meta = comm.irecv(self.buff, source=MPI.ANY_SOURCE).wait(status=state)
            total += num_samples
            log.info('Incoming grads from [%d], num_samples [%d]', state.Get_source(), num_samples)
            with self.work as ww: ww.on_result(meta)
            for ngrd,grd in zip(ngrads,grads): np.add(grd, ngrd, out=grd)
        # log.info('MASTER EXIT [%d]', total)
        for grd in grads: np.divide(grd, total, out=grd)

    def broadcast_vars(self, *vars):
        reqs = [comm.isend(vars, dest=i+1) for i in range(num_workers())]
        self.reqman.addn(reqs)
        return vars


class AsynchronousWorker:
    def __init__(self):
        self.reqman = ReqManMax1('DISPATCH')

    def dispatch_grads(self, *args):
        req = comm.isend(args, dest=ROOT_RANK)
        self.reqman.add(req)

    def update_vars(self, *vars):
        ret, count = vars, 0
        # check for the latest update
        while comm.Iprobe(source=ROOT_RANK):
            ret = comm.recv(source=ROOT_RANK)
            count += 1
        if count>0:
            log.info('New update from master! Iprobe count [%d]', count)
        else:
            log.info('No update! Master lagging!')
        return ret




###############################################################################

class WorkerProfiler:
    def __init__(self, dump_freq, columns, work_dir=None):
        self.reset()
        self.step_count = 0
        self.dump_freq = dump_freq
        self.csv = None if (work_dir is None) else \
                   ut.file.CSVFile('worker_stats.csv', work_dir, \
                           ['time', 'step'] + columns + ['compute_time_master'], mode='w')

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
        self.cache.append([elapsed_total, self.step_count] + list(data) + [compute_time_master])

    def __exit__(self, type, value, traceback):
        if self.step_count%self.dump_freq==0: self.dump_all()

    def dump_all(self):
        if self.csv is not None:
            for it in self.cache: self.csv.writerow(it)
            self.csv.flush()
            self.reset()

    def __del__(self): self.dump_all()
