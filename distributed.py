import tensorflow as tf
from mpi4py import MPI
import numpy as np
import logging
import socket
import time
import os

import utils


comm = MPI.COMM_WORLD
logger = logging.getLogger('dstr') # __name__
MPI_RANK = None
MPI_SIZE = None
WORK_DIR = None

def init(work_dir=None, log_level=logging.INFO): # INFO DEBUG

    global MPI_RANK, MPI_SIZE, WORK_DIR
    MPI_RANK = comm.Get_rank()
    MPI_SIZE = comm.Get_size()

    WORK_DIR = work_dir
    if rank()==0 and not os.path.exists(WORK_DIR): os.mkdir(WORK_DIR)

    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
            '%(asctime)s---------------------|%(name)s|%(message)s',
            # logging.BASIC_FORMAT,
            "%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    hostname = socket.gethostname()
    host = socket.gethostbyname(hostname)
    logger.info('Initialized rank [%d], hostname [%s], host [%s]', rank(), str(hostname), str(host))


def local_rank(): return 0
def rank(): return MPI_RANK
def size(): return MPI_SIZE
def num_workers(): return size()-1


def default_bcast_func(root_rank, *data):
    logger.debug('Broadcast from [%d], rank [%d]', root_rank, rank())
    return comm.bcast(data, root=root_rank)

def broadcast(tensors, root_rank, func):
    return tf.py_func(func=func, inp=[root_rank] + list(tensors),
                      Tout=tuple(tensor.dtype for tensor in tensors))

def broadcast_assign_vars(vars, root_rank, func=default_bcast_func):
    nvars = broadcast(vars, root_rank, func)
    return tf.group(*[tf.assign(var, nvar) for var, nvar in zip(vars, nvars)])


class BroadcastGlobalVariablesHook(tf.train.SessionRunHook):
    def __init__(self, root_rank, device=''):
        super(BroadcastGlobalVariablesHook, self).__init__()
        self.root_rank = root_rank
        self.bcast_op = None
        self.device = device

    def begin(self):
        if not self.bcast_op or self.bcast_op.graph != tf.get_default_graph():
            with tf.device(self.device):
                self.bcast_op = broadcast_assign_vars(tf.global_variables(), self.root_rank)

    def after_create_session(self, session, coord):
        session.run(self.bcast_op)


class MasterWorkerOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer, worker, straggler_stddev_ms, name=None, use_locking=False):

        self._optimizer = optimizer
        self.worker = worker
        self.wgrads, self.num_batches = None, 0
        self.std_dev = straggler_stddev_ms

        self.log = logger.getChild('MST' if rank()==0 else 'wk%d'%rank())
        self.prof = utils.LoopProfiler(self.log.getChild('LP'), 10000)
        self.work = utils.WorkerProfiler(WORK_DIR, self.log.getChild('WP'), 5)
        super(MasterWorkerOptimizer, self).__init__(name=name, use_locking=use_locking)

    def compute_gradients(self, *args, **kwargs):
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        if size() > 1:
            grads, vars = zip(*gradients)
            # TODO: test --> grads shouldn't be evaluated in master!!!
            shapes = tuple(grad.shape for grad in grads)
            Tout = tuple(grad.dtype for grad in grads)
            self.wgrads = list(np.zeros(ss, dtype=tt.as_numpy_dtype) for ss,tt in zip(shapes,Tout))

            func, inp = (self.master_func, []) if rank()==0 else (self.worker_func, grads)
            new_grads = tf.py_func(func=func, inp=inp, Tout=Tout)
            return list(zip(new_grads, vars))
        else:
            return gradients

    def apply_gradients(self, *args, **kwargs):
        apply_op = self._optimizer.apply_gradients(*args, **kwargs)
        if size()>1:
            # apply_op is created in workers only to create identical setup (global vars) to master.
            # TODO: test --> apply_op should never run in worker!!!! this is ensured by control_dependencies.
            grads, vars = zip(*args[0])
            if rank()==0:
                with tf.control_dependencies([apply_op]):
                    return broadcast_assign_vars(vars, 0, func=self.master_bcast_func)
            else:
                def f_true():
                    disp_op = tf.py_func(func=self.worker_dispatch_func, inp=[], Tout=[])
                    with tf.control_dependencies([disp_op]):
                        return broadcast_assign_vars(vars, 0)

                with tf.control_dependencies(grads):
                    is_dispatch_now = tf.py_func(func=self.worker.is_dispatch_now, inp=[], Tout=tf.bool)
                    return tf.cond(is_dispatch_now, f_true, tf.no_op)
        else:
            return apply_op


    def get_slot(self, *args, **kwargs):
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        return self._optimizer.variables(*args, **kwargs)

    # called only in the master node
    def master_func(self):
        log = self.log
        self.reset_grads()
        total_batches, recv_workers = 0, 0

        with self.prof as pp, self.work as ww:
            log.debug('Listening to [%d] workers', num_workers())
            with pp.tag('listen'):
                while recv_workers < num_workers():
                    # TODO: receive dont create new numy arrays.
                    # Mpi data type, send numpy arrays instead of python pickles
                    # https://mpi4py.readthedocs.io/en/stable/tutorial.html
                    # Try isend ie without wait
                    rank, summed_grads, num_batches, elapsed_worker = comm.recv()
                    self.add_grads(summed_grads)
                    ww.on_result(rank, elapsed_worker)
                    total_batches += num_batches
                    recv_workers += 1
                    log.debug('New grad! num_batches=[%d], total_batches=[%d]', num_batches, total_batches)

            with pp.tag('average'):
                return list(aa/total_batches for aa in self.wgrads)

    def master_bcast_func(self, root_rank, *data):
        with self.prof as pp:
            with pp.tag('broadcast'):
                return comm.bcast(data, root=root_rank)

    # def worker_bcast_func(self, root_rank, *data):
        # return comm.bcast(data, root=root_rank)

    def add_grads(self, grads):
        for i, arr in enumerate(self.wgrads):
            self.wgrads[i] = np.add(arr,grads[i], out=arr)

    def reset_grads(self):
        for i, arr in enumerate(self.wgrads):
            self.wgrads[i] = np.multiply(arr,0., out=arr)

    # called only in the worker nodes
    def worker_func(self, *grads):
        # # tt = abs(np.random.normal(0., self.std_dev))/1000.
        # # log.debug('Inducing straggler sleeping for [%g] ms.', tt)
        # # time.sleep(tt) # induced straggler sleeping time
        self.log.debug('Computed new_batch, num_batches [%d]', self.num_batches)
        self.worker.on_new_batch()
        self.add_grads(grads)
        self.num_batches += 1
        # TODO: REMOVE THIS RETURN!!!!
        return grads

    def worker_dispatch_func(self):
        self.log.debug('Sending summed_grads for [%d] batches', self.num_batches)
        comm.send((rank(), self.wgrads, self.num_batches, self.worker.elapsed()), dest=0)
        self.num_batches = 0
        self.reset_grads()
        self.worker.on_dispatch_done()


class WorkerTimer:
    def __init__(self): self.reset()
    def reset(self): self.started = time.time()
    def elapsed(self): return (time.time() - self.started)*1000

class FMBWorker(WorkerTimer):
    def __init__(self, every_n_batches):
        self.every_n_batches, self.batches_count = every_n_batches, 0
        super().__init__()
    def on_new_batch(self): self.batches_count += 1
    def is_dispatch_now(self): return self.batches_count >= self.every_n_batches
    def on_dispatch_done(self):
        self.batches_count = 0
        self.reset()

class FixedMiniBatchOptimizer(MasterWorkerOptimizer):
    def __init__(self, optimizer, every_n_batches, straggler_stddev_ms, name=None, use_locking=False):
        if name is None: name = "FixedMiniBatch{}".format(type(optimizer).__name__)
        super(FixedMiniBatchOptimizer, self).__init__(optimizer,
            FMBWorker(every_n_batches), straggler_stddev_ms, name=name, use_locking=use_locking)

class AnytimeWorker(WorkerTimer):
    def __init__(self, time_limit):
        self.limit, self.start = time_limit, None
        self.start = time.time()
        super().__init__()
    def on_new_batch(self): pass
    def is_dispatch_now(self): return (time.time() - self.start) >= self.limit
    def on_dispatch_done(self):
        self.start = time.time()
        self.reset()

class AnytimeOptimizer(MasterWorkerOptimizer):
    def __init__(self, optimizer, time_limit, straggler_stddev_ms, name=None, use_locking=False):
        if name is None: name = "Anytime{}".format(type(optimizer).__name__)
        super(AnytimeOptimizer, self).__init__(optimizer,
            AnytimeWorker(time_limit), straggler_stddev_ms, name=name, use_locking=use_locking)
