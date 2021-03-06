import tensorflow as tf
import numpy as np
import collections
import logging
import time
import os

import utilities as ut
import strategy as sgy

log = None


def safe_assign_vars(vars, func, arg=None, pred=None, pass_args=True):
    # py_func doesn't work on GPUs without casting
    def true_fn():
        vars_cast = [tf.cast(tt, tf.float64) for tt in vars]
        if pass_args: inputs = vars_cast if arg is None else (arg, *vars_cast)
        else: inputs = []
        nvars = tf.py_func(func=func, inp=inputs, Tout=tuple(tt.dtype for tt in vars_cast))
        nvars = [tf.cast(tt, tf.float32) for tt in nvars]
        return tf.group(*[tf.assign(var, nvar) for var, nvar in zip(vars, nvars)])

    def false_fn(): return tf.no_op()

    if pred is None:
        return true_fn()
    else:
        pred_op = tf.py_func(func=pred, inp=[], Tout=tf.bool)
        return tf.cond(pred_op, true_fn=true_fn, false_fn=false_fn)


class BroadcastVariablesHook(tf.train.SessionRunHook):
    def __init__(self, dist, device=''):
        super().__init__()
        self.vars = dist.get_variables()
        self.bcast_op = None
        self.device = device

    def begin(self):
        if not self.bcast_op or self.bcast_op.graph != tf.get_default_graph():
            with tf.device(self.device):
                self.bcast_op = safe_assign_vars(self.vars, sgy.default_bcast_func)

    def after_create_session(self, session, coord):
        session.run(self.bcast_op)


from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element

class CSVLoggingHook(session_run_hook.SessionRunHook):
    def __init__(self, every_n_iter, train_tensors, test_tensors={}, get_test_fd=None, flush_every_n_iter=10, work_dir=None):
        self._train_dict = train_tensors
        self._test_dict = test_tensors
        self._get_test_fd = get_test_fd
        self._every_n_iter = every_n_iter
        self._flush_every_n_iter = flush_every_n_iter
        self._tag_order = list(self._train_dict.keys()) + list(self._test_dict.keys())
        self._csv = None if work_dir is None else \
                    ut.file.CSVFile('master_stats.csv', work_dir, ['time'] + self._tag_order, mode='w')

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

def py_exc(expr):
    def inner(): expr()
    # expr is supposed to be a lambda. even though lambda
    # always returns something, inner returns nothing as expected by tf.py_func.
    return tf.py_func(func=inner, inp=[], Tout=[])

def ctrl_pyfunc(func, inp, Tout):
    op = tf.py_func(func=func, inp=inp, Tout=Tout)
    return tf.control_dependencies([op])

def log_d(fmt, *args):
    op = tf.py_func(func=log.debug, inp=[fmt]+[*args], Tout=[])
    return tf.control_dependencies([op])


class Distributor:
    def __init__(self, node):
        global log
        log = sgy.log
        self._node = node

    def minimize(self, placeholders, cr_sum_loss, global_step):
        self.vars, apply_op = self._node.minimize(placeholders, cr_sum_loss, global_step)
        return apply_op

    def get_variables(self):
        return self.vars

    def set_straggler(self, induce_dist):
        return self._node.set_straggler(induce_dist)

    def set_strategy(self, strategy):
        strategy.set_stats(stat_names)
        return self._node.set_strategy(strategy)


stat_names = ('last_send', 'last_recv', 'last_exit', 'sleep_time', 'compute_time')

class Master:#(tf.train.Optimizer):
    def __init__(self, optimizer, weight_decay):
        self._optimizer = optimizer
        self.weight_decay = weight_decay

    def set_straggler(self, *args): pass
    def set_strategy(self, strategy): self.strategy = strategy.make_master()

    def minimize(self, placeholders, cr_sum_loss, global_step):
        shapes, grads_and_vars = self.compute_gradients(placeholders, cr_sum_loss, global_step)
        sgy.default_bcast_func(*shapes)   ## send the variable shapes to workers
        _, vars = zip(*grads_and_vars)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            return vars, self.apply_gradients(grads_and_vars, global_step)

    def compute_gradients(self, placeholders, cr_sum_loss, global_step):
        grads_and_vars = comp_grads_total_loss(self._optimizer, cr_sum_loss(*placeholders), self.weight_decay)
        grads, vars = zip(*grads_and_vars)
        if sgy.size() > 1:
            shapes, Tout = zip(*[(grad.shape.as_list(), grad.dtype) for grad in grads])
            self.wgrads = list(np.zeros(ss, dtype=tt.as_numpy_dtype) for ss,tt in zip(shapes,Tout))
            new_grads = tf.py_func(func=self.collect_grads_func, inp=[global_step], Tout=Tout)
        else:
            batch_size = tf.dtypes.cast(tf.shape(placeholders[0])[0], tf.float32)
            new_grads = [grad/batch_size for grad in grads]
            shapes = []
        return shapes, list(zip(new_grads, vars))

    def apply_gradients(self, grads_and_vars, global_step):
        apply_op = self._optimizer.apply_gradients(grads_and_vars, global_step)
        if sgy.size()>1:
            grads, vars = zip(*grads_and_vars)
            with tf.control_dependencies([apply_op]):
                return safe_assign_vars(vars, self.strategy.send_update, arg=global_step)
        else:
            return apply_op

    def collect_grads_func(self, step):
        for arr in self.wgrads: np.multiply(arr, 0., out=arr)
        self.strategy.collect_grads(step, self.wgrads)
        return self.wgrads


class Straggler:
    def __init__(self, induce_dist, max_sleep_time):
        self.sleep_time = 0.
        means, std_devs, weights = zip(*induce_dist)
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


class WorkerTag:
    def __init__(self):
        self.tags = collections.OrderedDict()
        self.started = time.time()
        self.curr = ''

    def tag(self, name):
        now = time.time()
        elapsed = now-self.started
        self.tags[self.curr] = elapsed
        self.started = now
        self.curr = name
        return elapsed

    def get(self, name):
        return self.tags.get(name, -1)


class Worker:
    def __init__(self, optimizer, weight_decay):
        self._optimizer = optimizer
        self.weight_decay = weight_decay
        self.prof = WorkerTag()
        self.straggler = None

    def set_straggler(self, induce_dist):
        self.straggler = Straggler(induce_dist, 2**10)
        # super().__init__(name=self.__class__.__name__, use_locking=False)

    def reset(self): self.computing_started = time.time()
    def elapsed(self): return time.time() - self.computing_started
    def on_new_round_func(self):
        self.reset()
        self.prof.tag('compute')
        if self.straggler: self.straggler.induce()

    def set_strategy(self, strategy): self.strategy = strategy.make_worker()

    '''
    worker cycle:
        send the old grads (last_send)
        wait till master sends update (last_bcast)
        apply new parameters
        start the new round after this
        sleep if inducing stragglers (sleep_time)
        compute new grads (compute_time also includes sleep_time)
        send the new grads (the time for this will be sent with the next update)
    '''
    def send_grads_func(self, step, num_samples, *grads):
        # self.log.debug('Sending summed_grads for [%d] batches', num_batches)
        last_send = self.prof.get('send')
        last_recv = self.prof.get('recv')
        last_exit = self.prof.get('exit')
        sleep_time = self.straggler.sleep_time if self.straggler else 0.
        compute_time = self.prof.tag('send')
        # should be in the same order as stat_names variable
        stats = (last_send, last_recv, last_exit, sleep_time, compute_time)
        self.strategy.send_grads(step, grads, num_samples, stats)
        self.prof.tag('recv')

    def minimize(self, placeholders, cr_sum_loss, global_step):
        shapes = sgy.default_bcast_func(None) ## get the variable shapes from master
        # this is necessary b/c as of now in AMBworker code cr_sum_loss() can only be
        # executed after knowing the shapes of the accumilating variables
        with ctrl_pyfunc(self.on_new_round_func, [], []):
            grads_and_vars, num_samples = self.compute_gradients(shapes, placeholders, cr_sum_loss)
            grads, vars = zip(*grads_and_vars)
            with ctrl_dep(grads):
                return vars, self.apply_gradients(grads_and_vars, num_samples, global_step)

    def make_assign_op(self, vars, is_end_computation):
        pred_func = lambda: self.strategy.is_update_ready(is_end_computation)
        return safe_assign_vars(vars, self.strategy.receive_update, pred=pred_func, pass_args=False)

    def apply_gradients(self, grads_and_vars, num_samples, global_step):
        grads, vars = zip(*grads_and_vars)
        with ctrl_pyfunc(self.send_grads_func, inp=(global_step, num_samples, *grads), Tout=[]):
            with ctrl_dep([self.make_assign_op(vars, True)]):
                with ctrl_dep([tf.assign_add(global_step, 1)]):
                    return py_exc(lambda:self.prof.tag('exit'))

    def comp_grads_total_loss(self, sum_loss):
        return comp_grads_total_loss(self._optimizer, sum_loss, self.weight_decay)

def comp_grads_total_loss(optimizer, sum_loss, weight_decay):
    conv_var = [var for var in tf.trainable_variables()]
    # print(*conv_var, sep='\n'), exit()
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in conv_var])
    total_loss = l2_loss*weight_decay + sum_loss
    return optimizer.compute_gradients(total_loss)


class FixedMiniBatchWorker(Worker):
    def compute_gradients(self, _, placeholders, cr_sum_loss):
        batch_size = tf.shape(placeholders[0])[0]
        return self.comp_grads_total_loss(cr_sum_loss(*placeholders)), batch_size


class AnytimeMiniBatchWorker(Worker):
    def init(self, time_limit, num_partitions):
        self._time_limit, self._num_partitions = time_limit, num_partitions
        return self

    def compute_gradients(self, shapes, placeholders, cr_sum_loss):

        time_limit, num_partitions = self._time_limit, self._num_partitions
        batch_size = tf.shape(placeholders[0])[0]
        partition_size = tf.dtypes.cast(batch_size/num_partitions, tf.int32)

        def cond(curr_partition, *accs):
            #with log_d('Condition'):
            prnt = lambda: log.warn('Increase batch_size or decrease the amb_time_limit!')
            warn = lambda: tf.py_func(func=prnt, inp=[], Tout=[])
            okay = lambda: tf.no_op()
            chck = lambda: self.elapsed() < time_limit
            # check if already have gone through all the slipts, if this happens then should increase the batch size
            start_, end_ = start_end(curr_partition)
            # do at least one partition, otherwise will be sending all zeros
            chk0 = tf.equal(curr_partition, 0)
            chk1 = tf.shape(placeholders[0][start_:end_])[0] > 0
            chk2 = tf.py_func(func=chck, inp=[], Tout=tf.bool)
            with tf.control_dependencies([tf.cond(chk1, okay, warn)]):
                return tf.math.logical_or(chk0, tf.math.logical_and(chk1, chk2))

        def start_end(curr_partition):
            start_ = curr_partition*partition_size
            end_ = start_ + partition_size
            return start_, end_

        def body(curr_partition, *accs):
            start_, end_ = start_end(curr_partition)
            sum_loss = cr_sum_loss(*(pl[start_:end_] for pl in placeholders))
            grads_and_vars = self.comp_grads_total_loss(sum_loss)
            grads, self.vars = zip(*grads_and_vars)
            # print(list(ss.get_shape().as_list() for ss in self.vars))
            ret_accs = list(acc+grad for acc,grad in zip(accs, grads))
            # log_op = tf.py_func(func=log_, inp=[loss, curr_partition], Tout=[])
            assign_op = self.make_assign_op(self.vars, False)
            with tf.control_dependencies([*ret_accs, assign_op]):
                return [curr_partition+1] + ret_accs

        accs_0 = list(tf.zeros(shape) for shape in shapes)
        completed_partitions, *grads = tf.while_loop(cond, body, [tf.constant(0)] + accs_0,
                         parallel_iterations=1, return_same_structure=True, swap_memory=True)
        return list(zip(grads, self.vars)), completed_partitions*partition_size


class FixedMiniBatchDistributor(Distributor):
    def __init__(self, optimizer, weight_decay):
        obj = Master if sgy.rank()==0 else FixedMiniBatchWorker
        super().__init__(obj(optimizer, weight_decay))


class AnytimeMiniBatchDistributor(Distributor):
    def __init__(self, optimizer, weight_decay, *args, **kwargs):
        super().__init__(Master(optimizer, weight_decay) if sgy.rank()==0 else
                         AnytimeMiniBatchWorker(optimizer, weight_decay).init(*args, **kwargs))
