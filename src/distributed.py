import tensorflow as tf
import numpy as np
import logging
import socket
import time
import os

import utils


MPI = None
comm = None
log = None
ROOT_RANK = 0
MPI_RANK = None
MPI_SIZE = None
WORK_DIR = None
INDUCE_STRAGGLERS = None
INDUCE_DIST = None

def set_work_dir(work_dir=None):
    global WORK_DIR
    WORK_DIR = work_dir

def init(work_dir=None, induce_stragglers=0, induce_dist=None, log_level=logging.INFO): # INFO DEBUG

    global MPI, comm, log, MPI_RANK, MPI_SIZE, WORK_DIR, INDUCE_STRAGGLERS, INDUCE_DIST

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

    from mpi4py import MPI as MPI_
    MPI = MPI_
    comm = MPI.COMM_WORLD

    MPI_RANK = comm.Get_rank()
    MPI_SIZE = comm.Get_size()
    INDUCE_STRAGGLERS = induce_stragglers
    INDUCE_DIST = induce_dist
    WORK_DIR = work_dir

    log = logger.getChild('wk%d'%rank())
    log.info('Initialized rank [%d], hostname [%s], host [%s]', rank(), str(hostname), str(host))


def local_rank(): return 0
def rank(): return MPI_RANK
def size(): return MPI_SIZE
def num_workers(): return size()-1


def default_bcast_func(root_rank, *data):
    log.debug('Broadcast from [%d], rank [%d]', root_rank, rank())
    return comm.bcast(data, root=root_rank)

def broadcast(tensors, root_rank):
    return tf.py_func(func=default_bcast_func, inp=[root_rank] + list(tensors),
                      Tout=tuple(tensor.dtype for tensor in tensors))

def broadcast_assign_vars(vars, root_rank):
    nvars = broadcast(vars, root_rank)
    return tf.group(*[tf.assign(var, nvar) for var, nvar in zip(vars, nvars)])


class BroadcastVariablesHook(tf.train.SessionRunHook):
    def __init__(self, vars, root_rank, device=''):
        super().__init__()
        self.vars = vars
        self.root_rank = root_rank
        self.bcast_op = None
        self.device = device

    def begin(self):
        if not self.bcast_op or self.bcast_op.graph != tf.get_default_graph():
            with tf.device(self.device):
                self.bcast_op = broadcast_assign_vars(self.vars, self.root_rank)

    def after_create_session(self, session, coord):
        session.run(self.bcast_op)


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
                    utils.CSVFile('master_stats.csv', WORK_DIR, ['time'] + self._tag_order)

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
                log.info(line)
            else:
                log.info('%s (%.3f sec)'%(line, time.time()-self._last_time))

            self._last_time = time.time()

        if self._csv is not None and self._iter_count%self._flush_every_n_iter==0:
            self._csv.flush()

        self._iter_count += 1


ctrl_dep = tf.control_dependencies

def ctrl_pyfunc(func, inp, Tout):
    op = tf.py_func(func=func, inp=inp, Tout=Tout)
    return tf.control_dependencies([op])

def log_d(fmt, *args):
    op = tf.py_func(func=log.debug, inp=[fmt]+[*args], Tout=[])
    return tf.control_dependencies([op])

def mpi_reduce_grads(grads, num_samples, compute_time):
    # https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/
    # https://nyu-cds.github.io/python-mpi/05-collectives/
    compute_times_ll = comm.gather(compute_time, root=0)
    num_samples_ll = comm.gather(num_samples, root=0)
    #total = comm.allreduce(num_samples, op=MPI.SUM)
    for acc in grads:
        # https://nyu-cds.github.io/python-mpi/05-collectives/
        comm.Reduce(acc*(0. if rank()==0 else 1.), acc, op=MPI.SUM, root=0)
    return compute_times_ll, num_samples_ll


class Distributor:
    def __init__(self, node):
        self._node = node

    def minimize(self, placeholders, model_fac, global_step):
        self.vars, apply_op = self._node.minimize(placeholders, model_fac, global_step)
        return apply_op

    def get_variables(self):
        return self.vars


class Master:#(tf.train.Optimizer):
    def __init__(self, optimizer):
        self._optimizer = optimizer
        self.work = utils.WorkerProfiler(5, ['rank', 'num_samples', 'compute_time_worker'], WORK_DIR)
        # super().__init__(name=self.__class__.__name__, use_locking=False)

    def minimize(self, placeholders, model_fac, global_step):
        grads_and_vars = self.compute_gradients(placeholders, model_fac)
        _, vars = zip(*grads_and_vars)
        return vars, self.apply_gradients(grads_and_vars, global_step)

    def compute_gradients(self, placeholders, model_fac):
        grads_and_vars = self._optimizer.compute_gradients(model_fac(*placeholders))
        if size() > 1:
            grads, vars = zip(*grads_and_vars)
            shapes, Tout = zip(*[(grad.shape, grad.dtype) for grad in grads])
            self.wgrads = list(np.zeros(ss, dtype=tt.as_numpy_dtype) for ss,tt in zip(shapes,Tout))
            new_grads = tf.py_func(func=self.master_func, inp=[], Tout=Tout)
            return list(zip(new_grads, vars))
        else:
            return grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step):
        apply_op = self._optimizer.apply_gradients(grads_and_vars, global_step)
        if size()>1:
            grads, vars = zip(*grads_and_vars)
            with tf.control_dependencies([apply_op]):
                return broadcast_assign_vars(vars, 0)
        else:
            return apply_op

    # called only in the master node
    def master_func(self):
        log.debug('Listening to [%d] workers', num_workers())
        self.map_grads(np.multiply, 0.)
        compute_times_ll, num_samples_ll = mpi_reduce_grads(self.wgrads, 0, 0)
        compute_times_ll, num_samples_ll = compute_times_ll[1:], num_samples_ll[1:]
        total = sum(num_samples_ll)
        for acc in self.wgrads: np.divide(acc, total, out=acc)
        with self.work as ww:
            for i in range(num_workers()):
                ww.on_result([i+1, num_samples_ll[i], compute_times_ll[i]])
        return self.wgrads

    def map_grads(self, func, arg):
        for i, arr in enumerate(self.wgrads):
            self.wgrads[i] = func(arr,arg, out=arr)


class Straggler:
    def __init__(self, max_sleep_time):
        self.sleep_time = 0.
        means, std_devs, weights = zip(*INDUCE_DIST)
        modes = list(zip(means, std_devs))
        lim = min(max_sleep_time, max([it[0]+it[1]*5 for it in modes]))
        log.info('[INDUCED STRAGGLERS] modes:%s, weights:%s', modes, weights)
        def gen_sleep_():
            ind = np.random.choice(len(weights), p=weights)
            return min(lim, abs(np.random.normal(*modes[ind])))
        self.gen_sleep = gen_sleep_

    def induce(self):
        self.sleep_time = self.gen_sleep()
        log.debug('Inducing straggler sleeping for [%g]s.', self.sleep_time)
        time.sleep(self.sleep_time) # induced straggler sleeping time in secs


class Worker:
    def __init__(self, optimizer, batch_size):
        self._optimizer = optimizer
        self._batch_size = batch_size
        self.prof = utils.WorkerTag()
        if INDUCE_STRAGGLERS: self.straggler = Straggler(2**10)
        # super().__init__(name=self.__class__.__name__, use_locking=False)

    def reset(self): self.computing_started = time.time()
    def elapsed(self): return time.time() - self.computing_started
    def on_new_round_func(self):
        self.reset()
        self.prof.tag('compute')
        if INDUCE_STRAGGLERS: self.straggler.induce()

    def dispatch_func(self, num_samples, *grads):
        # self.log.debug('Sending summed_grads for [%d] batches', num_batches)
        last_send = self.prof.get('send')
        compute_time = self.prof.tag('send')
        log.info('Sending [%d] examples, compute_time [%g], last_idle [%g], last_send [%g]',\
                  num_samples, compute_time, self.prof.get('idle'), last_send)
        mpi_reduce_grads(grads, num_samples, compute_time)
        self.prof.tag('idle')

    def minimize(self, placeholders, model_fac, global_step):
        with ctrl_pyfunc(self.on_new_round_func, [], []):
            grads_and_vars, num_samples = self.compute_gradients(placeholders, model_fac)
            grads, vars = zip(*grads_and_vars)
            with ctrl_dep(grads):
                return vars, self.apply_gradients(grads_and_vars, num_samples, global_step)

    def apply_gradients(self, grads_and_vars, num_samples, global_step):
        grads, vars = zip(*grads_and_vars)
        with ctrl_pyfunc(self.dispatch_func, inp=[num_samples]+list(grads), Tout=[]):
            return broadcast_assign_vars(vars, 0)


class FixedMiniBatchWorker(Worker):
    def compute_gradients(self, placeholders, model_fac):
        return self._optimizer.compute_gradients(model_fac(*placeholders)), self._batch_size


class AnytimeMiniBatchWorker(Worker):
    def init(self, time_limit, num_splits):
        self._time_limit, self._num_splits = time_limit, num_splits
        return self

    def compute_gradients(self, placeholders, model_fac):

        time_limit, num_splits, batch_size = self._time_limit, self._num_splits, self._batch_size
        split_size = int(batch_size/num_splits)

        def cond(curr_split, *accs):
            #with log_d('Condition'):
            return tf.py_func(func=lambda: self.elapsed()<time_limit, inp=[], Tout=tf.bool)
            # return curr_split < num_splits

        def body(curr_split, *accs):
            start_ = curr_split*split_size
            end_ = start_ + split_size
            loss = model_fac(*(pl[start_:end_] for pl in placeholders))
            grads_and_vars = self._optimizer.compute_gradients(loss)
            grads, self.vars = zip(*grads_and_vars)
            # print(list(ss.get_shape().as_list() for ss in self.vars))
            ret_accs = list(acc+grad for acc,grad in zip(accs, grads))
            # log_op = tf.py_func(func=log_, inp=[loss, curr_split], Tout=[])
            with tf.control_dependencies(ret_accs):
                return [curr_split+1] + ret_accs

        accs_0 = list(tf.zeros(shape) for shape in model_fac.get_var_shapes())
        completed_splits, *grads = tf.while_loop(cond, body, [tf.constant(0)] + accs_0,
                         parallel_iterations=1, return_same_structure=True, swap_memory=True)
        return list(zip(grads, self.vars)), completed_splits*split_size


class FixedMiniBatchDistributor(Distributor):
    def __init__(self, optimizer, batch_size):
        super().__init__(Master(optimizer) if rank()==0 else
                         FixedMiniBatchWorker(optimizer, batch_size))


class AnytimeMiniBatchDistributor(Distributor):
    def __init__(self, optimizer, batch_size, *args, **kwargs):
        super().__init__(Master(optimizer) if rank()==0 else
                         AnytimeMiniBatchWorker(optimizer, batch_size).init(*args, **kwargs))
