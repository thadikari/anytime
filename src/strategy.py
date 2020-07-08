import numpy as np
import logging
import pickle
import socket
import time

import utilities as ut
import utilities.file
import utilities.misc


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

class FactoryBase:
    def __init__(self, work_dir): self.work_dir = work_dir

class SynchronousFac(FactoryBase):
    def make_master(self): return SynchronousMaster(self)
    def make_worker(self): return SynchronousWorker(self)
    def set_stats(self, stat_names): self.stat_names = ('num_samples', *stat_names)


def default_bcast_func(*data):
    log.debug('Broadcast from [%d], rank [%d]', ROOT_RANK, rank())
    return comm.bcast(data, root=ROOT_RANK)

def mpi_reduce_grads(grads, num_samples, stats):
    stats_ll = comm.gather(stats, root=ROOT_RANK) # reference [4,5]
    total = comm.reduce(num_samples, op=MPI.SUM, root=ROOT_RANK)
    for acc in grads: # reference [6]
        comm.Reduce(acc*(0. if rank()==0 else 1.), acc, op=MPI.SUM, root=ROOT_RANK)
    return total, stats_ll


class MasterBase:
    def __init__(self, fac):
        self.work = WorkerProfiler(5, fac.stat_names, fac.work_dir)
        self.fac = fac

class WorkerBase:
    def __init__(self, fac):
        self.fac = fac

    def print_stats(self, stats):
        log.info(', '.join(f'{nn}:{vv:g}' for nn,vv in zip(self.fac.stat_names, stats)))


class SynchronousMaster(MasterBase):
    def collect_grads(self, step, grads):
        self.work.start(step)
        log.debug('Listening to [%d] workers', num_workers())
        total, stats_ll = mpi_reduce_grads(grads, 0., None)
        for acc in grads: np.divide(acc, total, out=acc)
        for i in range(num_workers()):
            # mpi gather always ensures the rank order?? Yes as per [4]
            rank = i+1   # skip master
            self.work.on_result(rank, stats_ll[rank])
        self.work.end(total)

    def send_update(self, _, *vars): return default_bcast_func(*vars)


class SynchronousWorker(WorkerBase):
    def send_grads(self, _, grads, num_samples, stats):
        stats = (num_samples, *stats)
        self.print_stats(stats)
        return mpi_reduce_grads(grads, num_samples, stats)
    ## do not update in middle of the computation
    def is_update_ready(self, is_end_compute): return is_end_compute
    def receive_update(self): return default_bcast_func(None)




###############################################################################

class AsynchronousFac(FactoryBase):
    def master_args(self, **kwargs):
        def make_master():
            receiver = self.make_receiver(MasterReceiver())
            mkwargs = {**kwargs, 'receiver': receiver}
            return AsynchronousMaster(self).init(**mkwargs)
        self.make_master = make_master
        return self

    def make_worker(self):
        receiver = self.make_receiver(WorkerReceiver())
        return AsynchronousWorker(self).init(receiver=receiver)

    def make_receiver(self, inner):
        return RecvStraggler(inner, self.delay) if self.delay>0 else inner

    def set_straggler(self, delay_std):
        self.delay = delay_std
        return self

    def set_stats(self, stat_names):
        self.stat_names = ('worker_step', 'worker_master_step', 'last_queued_update_count', 'num_samples', *stat_names)


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
    def on_result(self): self.count += 1
    def should_wait(self): return self.count < self.threshold


class AsyncMasterTime:
    def __init__(self, threshold): self.threshold = threshold
    def reset(self):
        self.count = 0
        self.time = time.time()
    def on_result(self):
        self.count += 1
        log.info('on_result [%g], count [%d]', time.time()-self.time, self.count)
    def should_wait(self):
        # wait for at least one message
        if self.count==0: return True
        else: return time.time()-self.time <= self.threshold


class RecvReq:
    def __init__(self, src):
        self.buff = bytearray(1<<26)
        self.src = src
        self.reset()

    def reset(self):
        self.req = comm.irecv(self.buff, source=self.src)
        # ngrads, stats = comm.irecv(self.buff, source=MPI.ANY_SOURCE).wait(status=state)


class MasterReceiver:
    def __init__(self):
        self.reqs = [RecvReq(i+1) for i in range(num_workers())]
        # it is possible to make RecvReq that listens to MPI.ANY_SOURCE
        # but for some reason runs slower than current approach
        self.last_pos = 0

    def get_if_ready(self):
        self.last_pos += 1
        if self.last_pos==len(self.reqs): self.last_pos = 0
        req = self.reqs[self.last_pos]
        flag, data = req.req.test()
        if flag:
            req.req.wait()
            req.reset()
            return data, self.last_pos+1 # rank 1 + the array position
        else:
            return None


class RecvStraggler:
    def __init__(self, receiver, delay_std):
        self.inner = receiver
        self.delay_std = delay_std
        self.queue = []

    def get_if_ready(self):
        msg = self.inner.get_if_ready()
        if not msg is None: self.add(msg)
        return self.try_get_next()

    def add(self, msg):
        delay = abs(np.random.normal()*self.delay_std)
        expire = time.time() + delay
        self.queue.append((expire, msg))
        # print('delay', delay, '   time,', time.time())
        # print('before', list(zip(*self.queue))[0])
        self.queue.sort(key=lambda it: it[0])
        # print('after ', list(zip(*self.queue))[0])

    def try_get_next(self):
        if len(self.queue)>0 and time.time()>self.queue[0][0]:
            return self.queue.pop(0)[1]
        else: return None


class AsynchronousMaster(MasterBase):
    def init(self, style, batch_min, time_limit, receiver):
        self.reqman = ReqMan('MASTER')
        if style=='batch': self.checker = AsyncMasterBatch(batch_min)
        elif style=='time': self.checker = AsyncMasterTime(time_limit)
        self.reqs = receiver
        self.checker.reset()
        self.work.start(0)
        return self

    def collect_grads(self, step, grads):
        total = 0
        while self.checker.should_wait():
            msg = self.reqs.get_if_ready()
            if not msg is None:
                (ngrads, num_samples, stats), rank = msg
                total += num_samples
                log.info('Incoming grads from [%d], num_samples [%d]', rank, num_samples)
                self.checker.on_result()
                self.work.on_result(rank, stats)
                for ngrd,grd in zip(ngrads,grads): np.add(grd, ngrd, out=grd)
            else:
                time.sleep(0.00001)  # 0.1 millisecond, break between rounds

        self.work.end(total)

        assert(total>0)
        for grd in grads: np.divide(grd, total, out=grd)
        self.checker.reset()
        self.work.start(step+1)

    def send_update(self, step, *vars):
        reqs = [comm.isend((step, vars), dest=i+1) for i in range(num_workers())]
        self.reqman.addn(reqs)
        return vars


class AsynchronousWorker(WorkerBase):
    def init(self, receiver):
        self.reqman = ReqManMax1('WORKER')
        self.receiver = receiver
        self.last_master_step = 0
        self.lquc = 0
        self.data_ready = None
        return self

    def send_grads(self, step, grads, num_samples, stats):
        stats = (step, self.last_master_step, self.lquc, num_samples, *stats)
        req = comm.isend((grads, num_samples, stats), dest=ROOT_RANK)
        self.reqman.add(req)
        self.print_stats(stats)

    def receive_update(self):
        ret = self.data_ready
        self.data_ready = None
        return ret

    def is_update_ready(self, is_end_compute):
        assert(self.data_ready==None)
        msg = self.receiver.get_if_ready()
        if msg is None:
            self.lquc = 0
            if is_end_compute: log.info('No update! Master lagging!')
            return False
        else:
            data, self.lquc = msg
            ss = '' if is_end_compute else ' in middle of AMB computation!'
            log.info('Master update%s! Queued count [%d]', ss, self.lquc)
            self.last_master_step, self.data_ready = data
            return True


class WorkerReceiver:
    def __init__(self):
        self.newreq = lambda: comm.irecv(bytearray(1<<26), source=ROOT_RANK)
        self.reqs = [self.newreq() for i in range(10)]

    def get_if_ready(self):
        ret_data, count = None, 0
        while 1:
            # assumption: self.reqs[i] is completed
            # before self.reqs[i+k] for any i,k
            flag, data = self.reqs[0].test()
            if flag:
                self.reqs.append(self.newreq())
                self.reqs.pop(0).wait()
                ret_data = data
                count += 1
            else: break

        if ret_data is None:
            assert(count==0)
            return None
        else:
            return ret_data, count


###############################################################################

class WorkerProfiler:
    def __init__(self, dump_freq, columns, work_dir=None):
        self.ref_time = time.time()
        self.dump_freq = dump_freq
        col_names = ('time', 'master_step', 'rank', *columns, 'master_wait_time')
        self.wcsv = ut.file.CSVFile('worker_stats.csv', work_dir, col_names, mode='w')
        col_n = ('step', 'worker_count', 'wait_time', 'total_samples')
        self.mcsv = ut.file.CSVFile('collect_stats.csv', work_dir, col_n, mode='w')

    def end(self, total):
        elapsed = time.time() - self.round_started
        ndata = (self.master_step, self.worker_count, elapsed, total)
        self.mcsv.writerow(ndata)
        if self.master_step%self.dump_freq==0:
            self.wcsv.flush()
            self.mcsv.flush()

    def start(self, master_step):
        self.master_step = master_step
        self.round_started = time.time()
        self.worker_count = 0

    def on_result(self, rank, stats):
        curr = time.time()
        elapsed_total = curr - self.ref_time
        master_wait_time = curr - self.round_started
        ndata = (elapsed_total, self.master_step, rank, *stats, master_wait_time)
        self.wcsv.writerow(ndata)
        self.worker_count += 1
