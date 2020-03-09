# -*- coding: future_fstrings -*-

from os.path import isdir
from mpi4py import MPI
import argparse, logging
import time, os, json
import numpy as np

import utils
import utilities.file


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
root = 0

def workers_to_master(arr):
    comm.Reduce(arr+rank, arr, op=MPI.SUM, root=root)
    #comm.Reduce([arr+rank, MPI.DOUBLE], [arr, MPI.DOUBLE], op=MPI.SUM, root=root)
    return arr

def master_to_workers(arr):
    comm.Bcast([arr, MPI.DOUBLE], root=root)
    return arr


def main(_a):
    data = np.random.rand(_a.len_arr)

    if rank==0:
        args = vars(_a)
        num_workers = comm.Get_size()-1
        args['num_workers'] = num_workers
        args['num_bytes'] = data.nbytes
        print('[Arguments]', args)

        run_id = f'bandwidth__{_a.len_arr}_{num_workers}'
        work_dir = os.path.join(_a.data_dir, run_id)
        if not isdir(work_dir): os.makedirs(work_dir)
        with open(os.path.join(work_dir, 'args.json'), 'w') as fp_:
            json.dump(args, fp_, indent=4)

        cols = ['last_send', 'last_bcast'] + utils.LoopProfiler.extra_cols
        csv = utilities.file.CSVFile('worker_stats.csv', work_dir, cols)
        prof = utils.LoopProfiler(print, None, csv, _a.dump_freq)
    else:
        prof = utils.LoopProfiler(None, None, None, 1e8)

    for _ in range(_a.num_steps):
        with prof:
            with prof.tag('last_send'):
                data = workers_to_master(data)
            with prof.tag('last_bcast'):
                data = master_to_workers(data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('len_arr', type=int)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--dump_freq', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default=utilities.file.resolve_data_dir('distributed'))
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
