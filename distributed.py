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
INDUCE_STRAGGLERS = None

def init(work_dir=None, induce_stragglers=0, log_level=logging.INFO): # INFO DEBUG

    global comm, logger, MPI_RANK, MPI_SIZE, WORK_DIR, INDUCE_STRAGGLERS

    logger = logging.getLogger('dstr') # __name__
    if logger.hasHandlers(): logger.propagate = 0
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
    INDUCE_STRAGGLERS = induce_stragglers

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


callable_after_create_session = None
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
        global callable_after_create_session
        if callable_after_create_session is not None:
            callable_after_create_session()


from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element

class CSVLoggingHook(session_run_hook.SessionRunHook):
    def __init__(self, every_n_iter, train_tensors, test_tensors={}, get_test_fd=None, flush_every_n_iter=10):
        self._train_dict = train_tensors
        self._test_dict = test_tensors
        self._get_test_fd = get_test_fd
        self._every_n_iter = every_n_iter
        self._flush_every_n_iter = flush_every_n_iter
        self._tag_order = list(self._train_dict.keys()) + list(self._test_dict.keys())
        self._csv = None if WORK_DIR is None else \
                    utils.CSVFile('stats.csv', WORK_DIR, ['time'] + self._tag_order)

    def begin(self):
        self._iter_count = 0
        self._last_time = None
        self._test_tensors = list(self._test_dict.values())
        self._train_tensors = {tag: _as_graph_element(tensor)
                               for (tag, tensor) in self._train_dict.items()}

    def before_run(self, run_context):
        if self._iter_count==0: self._start_time = time.time()
        self._should_trigger = (self._iter_count%self._every_n_iter==0)
        return SessionRunArgs(self._train_tensors) if self._should_trigger else None

    def after_run(self, run_context, run_values):
        if self._should_trigger:

            if self._test_tensors:
                test_vals = run_context.session.run(self._test_tensors,
                                                    feed_dict=self._get_test_fd())
            else:
                test_vals = []

            train_vals = run_values.results
            stats = list(map(str, [train_vals[key] for key in self._train_dict.keys()] + test_vals))
            if self._csv is not None: self._csv.writerow([time.time()-self._start_time] + stats, False)

            line = ', '.join(map(lambda a_,b_: '%s = %s'%(a_,b_), self._tag_order, stats))
            if self._last_time is None: #tf.get_logger()
                logger.info(line)
            else:
                logger.info('%s (%.3f sec)'%(line, time.time()-self._last_time))

            self._last_time = time.time()

        if self._csv is not None and self._iter_count%self._flush_every_n_iter==0:
            self._csv.flush()

        self._iter_count += 1


class MasterWorkerOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer, worker_fac, name=None, use_locking=False):
        self._optimizer = optimizer
        if rank()!=0: self.worker = worker_fac()

        self.log = logger.getChild('MST')
        self.prof = utils.LoopProfiler(self.log.getChild('LP'), 10000)
        self.work = utils.WorkerProfiler(5, ['rank', 'num_batches', 'compute_time_worker'], WORK_DIR)
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
                    disp_op = tf.py_func(func=self.worker.dispatch_func, inp=[self.num_batches]+self.grad_accs, Tout=[])
                    with tf.control_dependencies([disp_op]):
                        reset_ops = [tf.assign(self.num_batches, 0.)]+\
                                    [tf.assign(acc, acc*0.) for acc in self.grad_accs]+\
                                    [broadcast_assign_vars(vars, 0)]
                        with tf.control_dependencies(reset_ops):
                            return tf.py_func(func=self.worker.on_new_round, inp=[], Tout=[])

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
        self.map_grads(np.multiply, 0.)
        total_batches, recv_workers = 0, 0

        with self.prof as pp, self.work as ww:
            log.debug('Listening to [%d] workers', num_workers())
            with pp.tag('listen'):
                while recv_workers < num_workers():
                    # TODO: receive dont create new numy arrays.
                    # Mpi data type, send numpy arrays instead of python pickles
                    # https://mpi4py.readthedocs.io/en/stable/tutorial.html
                    # Try isend ie without wait
                    rank, summed_grads, num_batches, compute_time_worker = comm.recv()
                    self.add_grads(summed_grads)
                    ww.on_result([rank, num_batches, compute_time_worker])
                    total_batches += num_batches
                    recv_workers += 1
                    log.debug('New grad! num_batches=[%d], total_batches=[%d]', num_batches, total_batches)
            # log.info('[%g]', np.sum(np.abs(self.wgrads[-1]))/total_batches)
            with pp.tag('average'):
                self.map_grads(np.divide, total_batches)
                return self.wgrads

    def master_bcast_func(self, root_rank, *data):
        with self.prof as pp:
            with pp.tag('broadcast'):
                return comm.bcast(data, root=root_rank)

    def add_grads(self, grads):
        for i, arr in enumerate(self.wgrads):
            self.wgrads[i] = np.add(arr,grads[i], out=arr)

    def map_grads(self, func, arg):
        for i, arr in enumerate(self.wgrads):
            self.wgrads[i] = func(arr,arg, out=arr)


class GenericWorker:
    def __init__(self, max_sleep_time=1e9):
        self.sleep_time = 0.
        self.prof = utils.WorkerTag()
        self.log = logger.getChild('wk%d'%rank())
        self.reset()

        global callable_after_create_session
        callable_after_create_session = self.on_new_round

        # modes = [[0, 0], [0, .2], [2, .1], [4, .2]]
        # weights = [.4, .3, .2, .1]
        modes = [[0, .1], [1.5, .4], [4, .5]]
        weights = [.5, .4, .1]
        lim = min(max_sleep_time, max([it[0]+it[1]*5 for it in modes]))
        if INDUCE_STRAGGLERS:
            self.log.info('[INDUCED STRAGGLERS] modes:%s, weights:%s', modes, weights)
        def gen_sleep_():
            ind = np.random.choice(len(weights), p=weights)
            return min(lim, abs(np.random.normal(*modes[ind])))
        self.gen_sleep = gen_sleep_

    def reset(self): self.computing_started = time.time()
    def elapsed(self): return time.time() - self.computing_started
    def on_new_round(self):
        self.reset()
        self.prof.tag('compute')
        self.induce_straggler()

    def induce_straggler(self):
        if INDUCE_STRAGGLERS:
            self.sleep_time = self.gen_sleep()
            logger.debug('Inducing straggler sleeping for [%g]s.', self.sleep_time)
            time.sleep(self.sleep_time) # induced straggler sleeping time in secs

    def dispatch_func(self, num_batches, *summed_grads):
        # self.log.debug('Sending summed_grads for [%d] batches', num_batches)
        last_send = self.prof.get('send')
        compute_time = self.prof.tag('send')
        self.log.info('Sending! Slept [%g], sending [%d] batches, compute_time [%g], last_idle [%g], last_send [%g]', self.sleep_time, num_batches, compute_time, self.prof.get('idle'), last_send)
        comm.send((rank(), summed_grads, num_batches, compute_time), dest=0)
        self.on_dispatch_done()
        self.prof.tag('idle')

def worker_factory(tt, *args):
    def create(): return tt(*args)
    return create

class FMBWorker(GenericWorker):
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
            worker_factory(FMBWorker, every_n_batches), name=name, use_locking=use_locking)

class AnytimeWorker(GenericWorker):
    def __init__(self, time_limit_sec):
        self.limit = time_limit_sec
        super().__init__(self.limit*0.8)
    def is_dispatch_now(self):
        return self.elapsed() >= self.limit
    def on_dispatch_done(self):
        pass

class AnytimeOptimizer(MasterWorkerOptimizer):
    def __init__(self, optimizer, time_limit_sec, name=None, use_locking=False):
        if name is None: name = "Anytime{}".format(type(optimizer).__name__)
        super(AnytimeOptimizer, self).__init__(optimizer,
            worker_factory(AnytimeWorker, time_limit_sec), name=name, use_locking=use_locking)
