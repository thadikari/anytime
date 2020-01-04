from mpi4py import MPI
import numpy as np
import logging
import socket


comm = MPI.COMM_WORLD
MPI_RANK = comm.Get_rank()
MPI_SIZE = comm.Get_size()

log.info('Initialized rank [%d], hostname [%s], host [%s]', rank(), str(hostname), str(host))


def default_bcast_func(root_rank, *data):
    log.debug('Broadcast from [%d], rank [%d]', root_rank, rank())
    return comm.bcast(data, root=root_rank)

    
def step():
    time.sleep(0.5)
    arr = comm.Reduce(arr, arr, op=MPI.SUM, root=0)

for _ in range(10):
    step()