import tensorflow as tf
from mpi4py import MPI
import numpy as np
import logging
import time
import os
import socket

import utils


comm = MPI.COMM_WORLD
logger = logging.getLogger('dstr') # __name__
WORK_DIR = None

def init(work_dir=None, log_level=logging.INFO): # INFO DEBUG
    global WORK_DIR
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
    logger.info('Initialized rank [%d] hostname [%s] host [%s]', rank(), str(hostname), str(host))


def local_rank(): return 0
def rank(): return comm.Get_rank()
def size(): return comm.Get_size()
def num_workers(): return size()-1


def bcast_func(root_rank, *data):
    logger.debug('Broadcast from [%d]', root_rank)
    return comm.bcast(data, root=root_rank)

def broadcast(tensors, root_rank):
    return tf.py_func(func=bcast_func, inp=[root_rank] + tensors,
                      Tout=tuple(tensor.dtype for tensor in tensors))


class BroadcastGlobalVariablesHook(tf.train.SessionRunHook):
    def __init__(self, root_rank, device=''):
        super(BroadcastGlobalVariablesHook, self).__init__()
        self.root_rank = root_rank
        self.bcast_op = None
        self.device = device

    def begin(self):
        if not self.bcast_op or self.bcast_op.graph != tf.get_default_graph():
            with tf.device(self.device):
                vars = tf.global_variables()
                nvars = broadcast(vars, self.root_rank)
                self.bcast_op = tf.group(*[tf.assign(var, nvar)
                                           for var, nvar in zip(vars, nvars)])

    def after_create_session(self, session, coord):
        session.run(self.bcast_op)


class MasterWorkerOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer, worker, straggler_stddev_ms, name=None, use_locking=False):

        self._optimizer = optimizer
        self.worker = worker
        self.wgrads, self.num_batches = None, 0
        self.std_dev = straggler_stddev_ms

        self.log = logger.getChild('MST' if rank() == 0 else 'wk%d'%rank())
        self.prof = utils.LoopProfiler(self.log.getChild('LP'), 10000)
        self.work = utils.WorkerProfiler(WORK_DIR, self.log.getChild('WP'), 5)
        super(MasterWorkerOptimizer, self).__init__(name=name, use_locking=use_locking)

    def compute_gradients(self, *args, **kwargs):
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        if size() > 1:
            grads, vars = zip(*gradients)
            Tout = tuple(grad.dtype for grad in grads)
            func, inp = (self.master_func, []) if rank() == 0 else (self.worker_func, grads)
            new_grads = tf.py_func(func=func, inp=inp, Tout=Tout)
            return list(zip(new_grads, vars))
        else:
            return gradients

    def apply_gradients(self, *args, **kwargs):
        return self._optimizer.apply_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        return self._optimizer.variables(*args, **kwargs)

    # called only in the master node
    def master_func(self):
        log = self.log
        grads, total_batches = [], 0
        log.debug('Listening to [%d] workers', num_workers())

        with self.prof as pp, self.work as ww:
            with pp.tag('listen'):
                while len(grads) < num_workers():
                    rank, summed_grads, batches, elapsed_worker = comm.recv()
                    # TODO: Pre-allocate numpy arrays in compute_gradients
                    grads.append(summed_grads)
                    total_batches += batches
                    ww.on_result(rank, elapsed_worker)
                    log.debug('New grad! batches=[%d], total_batches=[%d]', batches, total_batches)

            with pp.tag('average'):
                # log.debug('All received! Averaging for total_batches=[%d]', total_batches)
                # instead of adjusting learning rate,
                # use [sum(aa)], not [sum(aa)/len(ll)]
                # TODO: sum(aa)/total_batches
                avg_grads = list(sum(aa)/total_batches for aa in zip(*grads))

            with pp.tag('broadcast'):
                return comm.bcast(avg_grads, root=0)

    # called only in the worker nodes
    def worker_func(self, *grads):
        # time.sleep(abs(np.random.normal(0., self.std_dev))/1000.) # induced straggler sleeping time
        log = self.log
        self.worker.on_new_batch()
        self.num_batches += 1

        # TODO: Pre-allocate numpy arrays in compute_gradients
        if self.wgrads is None:
            # first time this function is called
            self.wgrads = list(np.copy(aa) for aa in grads)
        else:
            self.wgrads = list(np.add(aa,bb,out=aa) for aa,bb in zip(self.wgrads, grads))

        if self.worker.is_dispatch_now():
            log.debug('Sending summed_grads for [%d] batches', self.num_batches)
            comm.send((rank(), self.wgrads, self.num_batches, self.worker.elapsed()), dest=0)

            log.debug('Listening for avg_grads')
            # TODO: Broadcast receive dont create new numy arrays.
            # Mpi data type, send numpy arrays instead of python pickles
            # https://mpi4py.readthedocs.io/en/stable/tutorial.html
            # Try isend ie without wait
            avg_grads = comm.bcast(None, root=0)
            log.debug('Received avg_grads!')

            self.wgrads = list(np.multiply(aa,0.,out=aa) for aa in self.wgrads)
            self.worker.on_dispatch_done()
            self.num_batches = 0
            return avg_grads
        else:
            # do not take a gradient step with this iteration
            # hack is to apply zero gradient
            return list(np.multiply(aa,0.,out=aa) for aa in grads)


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
