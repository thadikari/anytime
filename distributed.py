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


def bcast_func(root_rank, *data):
    log.debug('Broadcast from [%d], rank [%d]', root_rank, rank())
    return comm.bcast(data, root=root_rank)

def broadcast_assign_vars(vars):
    nvars = tf.py_func(func=bcast_func, inp=[ROOT_RANK] + list(vars),
                       Tout=tuple(var.dtype for var in vars))
    return tf.group(*[tf.assign(var, nvar) for var, nvar in zip(vars, nvars)])


class BroadcastGlobalVariablesHook(tf.train.SessionRunHook):
    def __init__(self, device=''):
        super(BroadcastGlobalVariablesHook, self).__init__()
        self.bcast_op = None
        self.device = device

    def begin(self):
        if not self.bcast_op or self.bcast_op.graph != tf.get_default_graph():
            with tf.device(self.device):
                self.bcast_op = broadcast_assign_vars(tf.global_variables())

    def after_create_session(self, session, coord):
        session.run(self.bcast_op)


from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element

class LoggingHook(session_run_hook.SessionRunHook):
    def __init__(self, every_n_steps, global_step, train_tensors, test_tensors={}, get_test_fd=None, flush_every_n_steps=10):
        self._train_dict = train_tensors
        self._test_dict = test_tensors
        self._get_test_fd = get_test_fd
        self._every_n_steps = every_n_steps
        self._global_step = global_step
        self._flush_every_n_steps = flush_every_n_steps
        self._tag_order = list(self._train_dict.keys()) + list(self._test_dict.keys())
        self._csv = None if WORK_DIR is None or ROOT_RANK!=rank() else \
                    utils.CSVFile('master_stats.csv', WORK_DIR, ['time'] + self._tag_order)

    def begin(self):
        self._start_time = time.time()
        self._should_trigger = True
        self._last_time = None

        self._step_tensors = {'_global_step': self._global_step}
        self._train_tensors = {tag: _as_graph_element(tensor)
                               for (tag, tensor) in self._train_dict.items()}
        self._train_tensors['_global_step'] = self._global_step
        self._test_tensors = list(self._test_dict.values())

    def before_run(self, run_context):
        args = self._train_tensors if self._should_trigger else self._step_tensors
        return SessionRunArgs(args)

    def after_run(self, run_context, run_values):
        curr = run_values.results['_global_step']
        if self._should_trigger:
            self.on_new_global_step(run_context.session, run_values.results)
            self._should_trigger = False
            self._last_time = time.time()
            self._last_step = curr

            if self._csv is not None and curr%self._flush_every_n_steps==0:
                self._csv.flush()

        if self._last_step != curr:
            self._should_trigger = True

    def on_new_global_step(self, sess, train_vals):
        if self._test_tensors:
            test_vals = sess.run(self._test_tensors, feed_dict=self._get_test_fd(rank(), size()))
        else:
            test_vals = []

        rank_stats = [train_vals[key] for key in self._train_dict.keys()] + test_vals
        all_stats = comm.gather(rank_stats, root=ROOT_RANK)

        if rank()==0:
            stats = list(sum(l_)/len(l_) for l_ in zip(*all_stats))
            if self._csv is not None: self._csv.writerow([time.time()-self._start_time] + stats, False)
            line = ', '.join(map(lambda a_,b_: '%s = %g'%(a_,b_), self._tag_order, stats))
            if self._last_time is None: #tf.get_logger()
                log.info(line)
            else:
                log.info('%s (%.3f sec)'%(line, time.time()-self._last_time))


class GenericOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer, max_sleep_time, name=None, use_locking=False):
        self.opt = optimizer
        self.prof = utils.WorkerProfiler(10,
                    ['num_batches', 'computation', 'communication'],
                    WORK_DIR if rank()==ROOT_RANK else None)
        if INDUCE_STRAGGLERS: self.straggler = Straggler(max_sleep_time)
        self.num_batches = tf.get_variable('num_batches', dtype=tf.int32, shape=(), initializer=tf.zeros_initializer(), trainable=False)
        super(GenericOptimizer, self).__init__(name=name, use_locking=use_locking)

    def compute_gradients(self, *args, **kwargs):
        if size() > 1:
            gradients = tf.train.Optimizer(False, 'dummy').compute_gradients(*args, **kwargs)
            grads_, vars_ = zip(*gradients)
            self.grad_accs = list(self.opt._zeros_slot(grad, 'accumulator', 'accumulator_op') for grad in grads_)

            is_new_round = tf.cond(tf.equal(self.num_batches, 0),
                           lambda: tf.py_func(func=self.on_new_round, inp=[], Tout=[]),
                           tf.no_op)
            with tf.control_dependencies([is_new_round]):
                grads, vars = zip(*self.opt.compute_gradients(*args, **kwargs))
                assert(vars_==vars)
                with tf.control_dependencies([tf.assign_add(self.num_batches, 1)]):
                    new_grads = [tf.assign_add(acc, grad) for grad, acc in zip(grads, self.grad_accs)]
                    return list(zip(new_grads, vars))
        else:
            return self.opt.compute_gradients(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        if size() > 1:
            grads, vars = zip(*grads_and_vars)
            def f_true():
                ngrads = tf.py_func(func=self.all_reduce_func,
                                   inp=[self.num_batches]+self.grad_accs,
                                   Tout=tuple(var.dtype for var in vars))
                apply_op = self.opt.apply_gradients(list(zip(ngrads, vars)), global_step, name)
                with tf.control_dependencies([apply_op]):
                    reset_ops = [tf.assign(self.num_batches, 0)]+\
                                [tf.assign(acc, acc*0.) for acc in self.grad_accs]
                    return tf.group(reset_ops)

            with tf.control_dependencies(grads):
                is_dispatch_now = tf.py_func(func=self.is_dispatch_now, inp=[], Tout=tf.bool)
                return tf.cond(is_dispatch_now, f_true, tf.no_op)
        else:
            return self.opt.apply_gradients(grads_and_vars, global_step, name)

    def get_slot(self, *args, **kwargs):
        return self.opt.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self.opt.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        return self.opt.variables(*args, **kwargs)

    def on_new_round(self):
        self.on_new_round_()
        self.prof.reset_begin('computation')
        if INDUCE_STRAGGLERS: self.straggler.induce()

    # https://mpi4py.readthedocs.io/en/stable/tutorial.html
    # http://pages.tacc.utexas.edu/~eijkhout/pcse/html/
    # https://nyu-cds.github.io/python-mpi/05-collectives/

    # https://github.com/erdc/mpi4py/blob/master/src/MPI/msgbuffer.pxi
    # https://github.com/erdc/mpi4py/blob/master/demo/reductions/test_reductions.py
    def all_reduce_func(self, num_batches, *grad_accs):
        self.prof.begin('communication')
        total = comm.allreduce(num_batches, op=MPI.SUM)
        log.debug('Sent grad_accs for [%s] batches, total [%s]', num_batches, total)
        # log.info('SS: %s'%grad_accs[0][0][0][0][:4])
        for acc in grad_accs:
            comm.Allreduce(MPI.IN_PLACE, acc, op=MPI.SUM)
            np.divide(acc, total, out=acc)
        # log.info('EE: %s'%grad_accs[0][0][0][0][:4])

        self.prof.tag('num_batches', num_batches)
        self.prof.tag('rank', rank())

        rank_stats = self.prof.end()
        all_stats = comm.gather(rank_stats, root=ROOT_RANK)
        if ROOT_RANK==rank(): self.prof.dump(all_stats)

        return grad_accs


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


class FixedMiniBatchOptimizer(GenericOptimizer):
    def __init__(self, optimizer, every_n_batches, name=None, use_locking=False):
        self.every_n_batches, self.batches_count = every_n_batches, 0
        if name is None: name = "FixedMiniBatch{}".format(type(optimizer).__name__)
        super(FixedMiniBatchOptimizer, self).__init__(optimizer,
            1e9, name=name, use_locking=use_locking)

    def is_dispatch_now(self):
        self.batches_count += 1
        return self.batches_count >= self.every_n_batches

    def on_new_round_(self):
        self.batches_count = 0


class AnytimeMiniBatchOptimizer(GenericOptimizer):
    def __init__(self, optimizer, time_limit_sec, name=None, use_locking=False):
        self.limit = time_limit_sec
        if name is None: name = "AnytimeMiniBatch{}".format(type(optimizer).__name__)
        super(AnytimeMiniBatchOptimizer, self).__init__(optimizer,
            self.limit*0.9, name=name, use_locking=use_locking)

    def is_dispatch_now(self): return self.elapsed() >= self.limit
    def on_new_round_(self): self.reset()
    def reset(self): self.computing_started = time.time()
    def elapsed(self): return time.time() - self.computing_started
