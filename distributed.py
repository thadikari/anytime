import tensorflow as tf
import numpy as np
import logging
import socket
import time
import os

import utils


comm = None
logger = None
MPI_RANK = None
MPI_SIZE = None
WORK_DIR = None
STRAGGLER_STD_DEV_MS = None

def init(work_dir=None, straggler_std_dev_ms=0, log_level=logging.INFO): # INFO DEBUG

    global comm, logger, MPI_RANK, MPI_SIZE, WORK_DIR, STRAGGLER_STD_DEV_MS

    logger = logging.getLogger('dstr') # __name__
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
    logger.info('Initializing...')

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    MPI_RANK = comm.Get_rank()
    MPI_SIZE = comm.Get_size()
    STRAGGLER_STD_DEV_MS = straggler_std_dev_ms

    WORK_DIR = work_dir
    if WORK_DIR is not None \
        and rank()==0 \
        and not os.path.exists(WORK_DIR):
            os.mkdir(WORK_DIR)

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


from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element

class CSVLoggingHook(session_run_hook.SessionRunHook):
    def __init__(self, tensors, every_n_iter, flush_every_n_iter=10):
        self._tensors = tensors
        self._every_n_iter = every_n_iter
        self._flush_every_n_iter = flush_every_n_iter
        self._tag_order = tensors.keys()
        self._csv = utils.CSVFile('stats.csv', WORK_DIR,
                                 ['time (sec)'] + list(self._tag_order))

    def begin(self):
        self._iter_count = 0
        self._start_time = time.time()
        self._current_tensors = {tag: _as_graph_element(tensor)
                                 for (tag, tensor) in self._tensors.items()}

    def before_run(self, run_context):
        self._should_trigger = (self._iter_count%self._every_n_iter==0)
        return SessionRunArgs(self._current_tensors) if self._should_trigger else None

    def after_run(self, run_context, run_values):
        if self._should_trigger:
            tensor_values = run_values.results
            stats = list('%s'%tensor_values[tag] for tag in self._tag_order)
            self._csv.writerow([time.time()-self._start_time] + stats, False)

        if self._iter_count%self._flush_every_n_iter==0:
            self._csv.flush()

        self._iter_count += 1


class MasterWorkerOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer, worker, name=None, use_locking=False):

        self._optimizer = optimizer
        self.worker = worker

        self.log = logger.getChild('MST' if rank()==0 else 'wk%d'%rank())
        self.prof = utils.LoopProfiler(self.log.getChild('LP'), 10000)
        self.work = utils.WorkerProfiler(5, WORK_DIR, self.log.getChild('WP'))
        super(MasterWorkerOptimizer, self).__init__(name=name, use_locking=use_locking)

    def compute_gradients(self, *args, **kwargs):
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        if size() > 1:
            grads, vars = zip(*gradients)
            # TODO: test --> grads shouldn't be evaluated in master!!!
            shapes, Tout = zip(*[(grad.shape, grad.dtype) for grad in grads])
            self.wgrads = list(np.zeros(ss, dtype=tt.as_numpy_dtype) for ss,tt in zip(shapes,Tout))
            self.num_batches = tf.get_variable('worker_num_batches', shape=(), initializer=tf.zeros_initializer(), trainable=False)
            self.grad_accs = list(self._optimizer._zeros_slot(grad, 'accumulator', 'accumulator_op') for grad in grads)
            # self.grad_accs created in master only to create identical setup (global vars) to master.

            if rank()==0:
                new_grads = tf.py_func(func=self.master_func, inp=[], Tout=Tout)
            else:
                with tf.control_dependencies([tf.assign_add(self.num_batches, 1)]):
                    new_grads = [tf.assign_add(acc, grad) for grad, acc in zip(grads, self.grad_accs)]

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
                    disp_op = tf.py_func(func=self.worker_dispatch_func, inp=[self.num_batches]+self.grad_accs, Tout=[])
                    with tf.control_dependencies([disp_op]):
                        reset_ops = [tf.assign(self.num_batches, 0.)]+\
                                    [tf.assign(acc, acc*0.) for acc in self.grad_accs]+\
                                    [broadcast_assign_vars(vars, 0)]
                        with tf.control_dependencies(reset_ops):
                            return tf.py_func(func=self.worker.on_computing_start, inp=[], Tout=[])

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
            # log.info('[%g]', np.sum(np.abs(self.wgrads[-1]))/total_batches)
            with pp.tag('average'):
                return list(aa/total_batches for aa in self.wgrads)

    def master_bcast_func(self, root_rank, *data):
        with self.prof as pp:
            with pp.tag('broadcast'):
                return comm.bcast(data, root=root_rank)

    def add_grads(self, grads):
        for i, arr in enumerate(self.wgrads):
            self.wgrads[i] = np.add(arr,grads[i], out=arr)

    def reset_grads(self):
        for i, arr in enumerate(self.wgrads):
            self.wgrads[i] = np.multiply(arr,0., out=arr)

    def worker_dispatch_func(self, num_batches, *summed_grads):
        # self.log.debug('Sending summed_grads for [%d] batches', num_batches)
        self.log.debug('Sending summed_grads. Slept [%g]ms, sending [%d] batches', self.worker.sleep_time, num_batches)
        comm.send((rank(), summed_grads, num_batches, self.worker.elapsed_ms()), dest=0)
        self.worker.on_dispatch_done()


class WorkerTimer:
    def __init__(self):
        self.last_idle, self.sleep_time = 0, 0
        self.reset()
    def reset(self): self.computing_started = time.time()
    def elapsed_ms(self): return (time.time() - self.computing_started)*1000
    def on_computing_start(self):
        self.last_idle = self.elapsed_ms()
        self.reset()
        self.induce_straggler()
    def induce_straggler(self):
        if STRAGGLER_STD_DEV_MS>0:
                # tt = abs(np.random.normal(0., STRAGGLER_STD_DEV_MS))
            modes = [[0, 0.], [0, 200], [2000, 100], [5000, 100]]
            weights = [.4, .3, .2, .1]
            ind = np.random.choice(len(weights), p=weights)
            self.sleep_time = abs(np.random.normal(*modes[ind]))
            logger.debug('Inducing straggler sleeping for [%g] ms.', self.sleep_time)
            time.sleep(self.sleep_time/1000.) # induced straggler sleeping time in secs

class FMBWorker(WorkerTimer):
    def __init__(self, every_n_batches):
        self.every_n_batches, self.batches_count = every_n_batches, 0
        super().__init__()
    def is_dispatch_now(self):
        self.batches_count += 1
        return self.batches_count >= self.every_n_batches
    def on_dispatch_done(self):
        self.batches_count = 0

class FixedMiniBatchOptimizer(MasterWorkerOptimizer):
    def __init__(self, optimizer, every_n_batches, name=None, use_locking=False):
        if name is None: name = "FixedMiniBatch{}".format(type(optimizer).__name__)
        super(FixedMiniBatchOptimizer, self).__init__(optimizer,
            FMBWorker(every_n_batches), name=name, use_locking=use_locking)

class AnytimeWorker(WorkerTimer):
    def __init__(self, time_limit_sec):
        self.limit_ms = time_limit_sec*1000
        super().__init__()
    def is_dispatch_now(self):
        return self.elapsed_ms() >= self.limit_ms
    def on_dispatch_done(self):
        pass

class AnytimeOptimizer(MasterWorkerOptimizer):
    def __init__(self, optimizer, time_limit_sec, name=None, use_locking=False):
        if name is None: name = "Anytime{}".format(type(optimizer).__name__)
        super(AnytimeOptimizer, self).__init__(optimizer,
            AnytimeWorker(time_limit_sec), name=name, use_locking=use_locking)
