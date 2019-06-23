import tensorflow as tf
from mpi4py import MPI
import sys, time
import logging


comm = MPI.COMM_WORLD
mylog = lambda l_: print('----------------------RANK[%d]----:%s'%(comm.Get_rank(), l_))
comlog = lambda l_: 0#mylog(l_)


def init(): mylog('init')

def local_rank(): return 0

def rank(): return comm.Get_rank()

def size(): return comm.Get_size()


def bcast_func(root_rank, *data):
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


def master_func(*grads):
    comlog('receiveing new_grads')
    new_grads = comm.recv(source=1)
    comlog('RECEIVED, sending update')
    comm.send(new_grads, dest=1)
    comlog('sent update')
    return new_grads

def worker_func(*grads):
    # print(type(grads), len(grads))
    comlog('sending worker')
    comm.send(grads, dest=0)
    comlog('sent grads, receiving update')
    new_grads = comm.recv(source=0)
    comlog('RECEIVED update!!!!')
    return new_grads


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/optimizer.py
# https://github.com/horovod/horovod/blob/master/horovod/tensorflow/__init__.py
class DistributedOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer, name=None, use_locking=False):
        if name is None: name = "Distributed{}".format(type(optimizer).__name__)
        self._optimizer = optimizer
        super(DistributedOptimizer, self).__init__(
            name=name, use_locking=use_locking)

    def compute_gradients(self, *args, **kwargs):
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        if size() > 1:
            func = master_func if rank() == 0 else worker_func
            grads, vars = zip(*gradients)
            new_grads = tf.py_func(func=func, inp=grads,
                Tout=tuple(grad.dtype for grad in grads))
            return list(zip(new_grads, vars))
        else:
            return gradients

    def apply_gradients(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.apply_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.variables(*args, **kwargs)
