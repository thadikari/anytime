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
    def make_master(self): return SynchronousMaster(self.work_dir)
    def make_worker(self): return SynchronousWorker()


def default_bcast_func(*data):
    log.debug('Broadcast from [%d], rank [%d]', ROOT_RANK, rank())
    return comm.bcast(data, root=ROOT_RANK)

def mpi_reduce_grads(grads, meta):
    meta_ll = comm.gather(meta, root=0) # reference [4,5]
    # total = comm.allreduce(num_samples, op=MPI.SUM)
    for acc in grads: # reference [6]
        comm.Reduce(acc*(0. if rank()==0 else 1.), acc, op=MPI.SUM, root=0)
    return meta_ll


class SynchronousMaster:
    def __init__(self, work_dir):
        self.work = WorkerProfiler(5, log_col_names, work_dir)

    def collect_grads(self, grads):
        log.debug('Listening to [%d] workers', num_workers())
        meta_ll = mpi_reduce_grads(grads, np.zeros(len(log_col_names)))
        meta_ll = [arr.tolist() for arr in meta_ll[1:]]
        total = sum(meta[0] for meta in meta_ll)
        for acc in grads: np.divide(acc, total, out=acc)
        with self.work as ww:
            # mpi gather always ensures the rank order??
            for i in range(num_workers()):
                ww.on_result(meta_ll[i])

    def broadcast_vars(self, *vars): return default_bcast_func(*vars)


class SynchronousWorker:
    def dispatch_grads(self, *args): return mpi_reduce_grads(*args)
    def update_vars(self, *vars): return default_bcast_func(*vars)




###############################################################################

class Asynchronous:
    def __init__(self, work_dir): self.work_dir = work_dir
    def make_master(self): return AsynchronousMaster(self.work_dir)
    def make_worker(self): return AsynchronousWorker()


class ReqMan:
    def __init__(self, name=None):
        self.name, self.reqs = name, []

    def add_clear(self, req):
        self.reqs.append(req)
        # test() returns (flag, msg) whereas Test() returns flag only. See reference [2].
        # if Test() is true the resources are automatically freed as per [3].
        # instead of storing requests, can also just free them as mentioned in [3].
        self.reqs = [req for req in self.reqs if not req.Test()]
        log.info('%s queue size [%d]', self.name if self.name else 'Requests', len(self.reqs))


class AsynchronousMaster:
    def __init__(self, work_dir):
        self.work = WorkerProfiler(5, log_col_names, work_dir)
        self.reqman = ReqMan('BCAST')
        # self.buff = bytearray(1<<30)

    def collect_grads(self, grads):
        state = MPI.Status()
        # ngrads, meta = comm.irecv(self.buff, source=MPI.ANY_SOURCE).wait(status=state)
        ngrads, meta = comm.recv(source=MPI.ANY_SOURCE, status=state)
        log.info('Incoming grads from [%d]', state.Get_source())
        for ngrd,grd in zip(ngrads,grads): np.divide(ngrd, meta[0], out=grd)
        with self.work as ww: ww.on_result(meta.tolist())

    def broadcast_vars(self, *vars):
        self.reqman.add_clear(comm.Ibcast(pickle.dumps(vars), root=ROOT_RANK))
        return vars


class AsynchronousWorker:
    def __init__(self):
        self.reqman = ReqMan('DISPATCH')
        self.buff = bytearray(2**30)
        self.new_req = lambda: comm.Ibcast(self.buff, root=ROOT_RANK)
        self.bcast_req = self.new_req()

    def dispatch_grads(self, grads, meta):
        req = comm.isend((grads, meta), dest=ROOT_RANK)
        self.reqman.add_clear(req)

    def update_vars(self, *vars):
        if self.bcast_req.Test():
            ret = pickle.loads(self.buff)
            self.bcast_req = self.new_req()
            log.info('New broadcast from master!!')
        else:
            ret = vars
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
        self.cache.append([elapsed_total, self.step_count] + data + [compute_time_master])

    def __exit__(self, type, value, traceback):
        if self.step_count%self.dump_freq==0: self.dump_all()

    def dump_all(self):
        if self.csv is not None:
            for it in self.cache: self.csv.writerow(it)
            self.csv.flush()
            self.reset()

    def __del__(self): self.dump_all()
