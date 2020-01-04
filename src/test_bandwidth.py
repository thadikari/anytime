# -*- coding: future_fstrings -*-

from os.path import isdir
from mpi4py import MPI
import argparse, logging
import time, os, json
import numpy as np

import utils


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
root = 0

def workers_to_master(arr):
    comm.Reduce(arr+rank, arr, op=MPI.SUM, root=root)
    return arr

def master_to_workers(arr):
    comm.Bcast([arr, MPI.DOUBLE], root=root)
    return arr


def main(_a):
    if rank==0:
        args = vars(_a)
        num_workers = comm.Get_size()-1
        args['num_workers'] = num_workers
        print('[Arguments]', args)

        run_id = f'bandwidth__{_a.barrier}_{_a.no_bcast}_{_a.len_arr}_{num_workers}'
        work_dir = os.path.join(_a.data_dir, run_id)
        if not isdir(work_dir): os.makedirs(work_dir)
        with open(os.path.join(work_dir, 'args.json'), 'w') as fp_:
            json.dump(args, fp_, indent=4)

        csv = utils.CSVFile('worker_stats.csv', work_dir, ['last_send', 'last_bcast'])
        prof = utils.LoopProfiler(print, None, csv, _a.dump_freq)
    else:
        prof = utils.LoopProfiler(None, None, None, 1e8)


    data = np.random.rand(_a.len_arr)
    for _ in range(_a.num_steps):
        with prof:
            with prof.tag('last_send'):
                data = workers_to_master(data)
                if _a.barrier: comm.Barrier()
            with prof.tag('last_bcast'):
                if not _a.no_bcast:
                    data = master_to_workers(data)
                    if _a.barrier: comm.Barrier()
            # print(rank, data)
            time.sleep(0.5)


def parse_args():
    SCRATCH = os.environ.get('SCRATCH', None)
    if not SCRATCH: SCRATCH = os.path.join(os.path.expanduser('~'), 'SCRATCH')
    data_dir = os.path.join(SCRATCH, 'distributed')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('len_arr', type=int)
    parser.add_argument('--no_bcast', action='store_true')
    parser.add_argument('--barrier', action='store_true')
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--dump_freq', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default=data_dir)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
