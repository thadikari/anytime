# -*- coding: future_fstrings -*-

import tensorflow as tf
import numpy as np
import argparse
import json
import os


import distributed as hvd
import cifar10
import mnist

tf.logging.set_verbosity(tf.logging.INFO)


def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('model', choices=['mnist', 'cifar10'])
    parser.add_argument('dist_opt', choices=['fmb', 'amb'])
    parser.add_argument('dist_param', type=float)
    parser.add_argument('batch_size', type=int)

    parser.add_argument('intr_opt', choices=['sgd', 'rms', 'adm'])

    parser.add_argument('--induce', help='induce stragglers', action='store_true')
    parser.add_argument('--dist', default=\
                                        # [[0, .1, 0.5], [1.5, .4, .4], [4, .5, .1]])
                                        # [[0, 0, .4], [0, .2, .3], [2, .1, .2], [4, .2, .1]]
                                        [[0, 0, .4], [0, .4, .3], [2, .3, .2], [5, .3, .1]]
                                        )

    parser.add_argument('--extra', default=None, type=str)
    parser.add_argument('--no_stats', help='do not save stats', action='store_true')
    parser.add_argument('--log_freq', default=1, type=int)
    return parser


def main(a):
    starter_learning_rate = 0.001
    decay_steps, decay_rate = 50, 0.96
    last_step = 1000000
    test_size = 100

    SCRATCH = os.environ.get('SCRATCH', '/home/sgeadmin')
    extra_line = '' if a.extra is None else '__%s'%a.extra
    run_id = f'{a.model}{extra_line}__{a.dist_opt}_{a.dist_param:g}_{a.batch_size}__{a.intr_opt}_{starter_learning_rate:g}_{decay_steps}_{decay_rate:g}__{a.induce}'
    print('run_id: %s'%run_id)
    logs_dir = None if a.no_stats else os.path.join(SCRATCH, 'checkpoints', run_id)

    # Horovod: initialize Horovod.
    hvd.init(induce_stragglers=a.induce, induce_dist=a.dist)
    if hvd.rank()==0:
        if logs_dir is not None:
            if not os.path.exists(logs_dir): os.mkdir(logs_dir)
            with open(os.path.join(logs_dir, 'args'), 'w') as fp_:
                json.dump(vars(a), fp_, indent=4)
    hvd.set_work_dir(logs_dir) 
    loss, accuracy, get_train_fd, get_test_fd = getattr(globals()[a.model], 'get_everything')(a.batch_size, test_size=test_size)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate,
                    global_step, decay_steps, decay_rate, staircase=True)

    # Horovod: adjust learning rate based on number of GPUs.
    # in anytime impl this adjustment is made internally, by the number of steps returned by each worker
    if a.intr_opt == 'rms': opt = tf.train.RMSPropOptimizer(learning_rate)#*max(1, hvd.size()-1))
    elif a.intr_opt == 'adm': opt = tf.train.AdamOptimizer(learning_rate)#*max(1, hvd.size()-1))
    elif a.intr_opt == 'sgd': opt = tf.train.GradientDescentOptimizer(learning_rate)

    # Horovod: add Horovod Distributed Optimizer.
    if a.dist_opt == 'fmb': opt = hvd.FixedMiniBatchOptimizer(opt, every_n_batches=int(a.dist_param))
    elif a.dist_opt == 'amb': opt = hvd.AnytimeMiniBatchOptimizer(opt, time_limit_sec=float(a.dist_param))

    train_op = opt.minimize(loss, global_step=global_step)
    hooks = [hvd.BroadcastGlobalVariablesHook(0),
             tf.train.StopAtStepHook(last_step=last_step),]# // hvd.size()),
    if hvd.rank()==0:
        hooks.append(hvd.CSVLoggingHook(every_n_iter=a.log_freq, 
                     train_tensors={'step':global_step, 'loss':loss, 'learning_rate':learning_rate}, 
                     test_tensors={'accuracy':accuracy}, get_test_fd=get_test_fd)) # 'accuracy':accuracy

    with tf.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op, feed_dict=get_train_fd())


if __name__ == '__main__': main(setup_parser().parse_args())
