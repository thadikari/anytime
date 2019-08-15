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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model', choices=['mnist', 'cifar10'])
    parser.add_argument('dist_opt', choices=['fmb', 'amb'])
    parser.add_argument('intr_opt', choices=['sgd', 'rms', 'adm'])
    parser.add_argument('batch_size', type=int)

    parser.add_argument('--amb_time_limit', type=float)
    parser.add_argument('--amb_num_splits', type=int)

    parser.add_argument('--induce', help='induce stragglers', action='store_true')
    parser.add_argument('--dist', default=\
                                        # [[0, .1, 0.5], [1.5, .4, .4], [4, .5, .1]])
                                        # [[0, 0, .4], [0, .2, .3], [2, .1, .2], [4, .2, .1]]
                                        [[0, 0, .4], [0, .4, .3], [2, .3, .2], [5, .3, .1]]
                                        )

    parser.add_argument('--starter_learning_rate', default=0.001, type=float)
    parser.add_argument('--decay_steps', default=50, type=int)
    parser.add_argument('--decay_rate', default=1, type=float)

    parser.add_argument('--extra', default=None, type=str)
    parser.add_argument('--no_stats', help='do not save stats', action='store_true')
    parser.add_argument('--log_freq', default=1, type=int)
    parser.add_argument('--last_step', default=1000000, type=int)
    parser.add_argument('--test_size', default=-1, type=int)
    args = parser.parse_args()

    vv = vars(args)
    if args.dist_opt=='amb':
        if args.amb_time_limit is None or args.amb_num_splits is None:
            parser.error('Need to define both amb_time_limit and amb_time_splis.')
        if args.amb_num_splits > args.batch_size:
            parser.error('Case not allowed: args.amb_num_splits > args.batch_sizes')

    return args


def main(a):
    SCRATCH = os.environ.get('SCRATCH', '/home/sgeadmin')
    extra_line = '' if a.extra is None else '__%s'%a.extra
    amb_args = f'__{a.amb_time_limit:g}_{a.amb_num_splits}'
    run_id = f'{a.model}{extra_line}__{a.dist_opt}_{a.intr_opt}_{a.batch_size}{amb_args}__{a.starter_learning_rate:g}_{a.decay_steps}_{a.decay_rate:g}__{a.induce}'
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

    model = globals()[a.model]
    placeholders, model_fac, get_train_fd, get_test_fd = model.get_fac_elements(a.batch_size, a.test_size)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.compat.v1.train.exponential_decay(a.starter_learning_rate,
                    global_step, a.decay_steps, a.decay_rate, staircase=True)

    # Horovod: adjust learning rate based on number of GPUs.
    # in anytime impl this adjustment is made internally, by the number of steps returned by each worker
    if a.intr_opt == 'rms': opt = tf.train.RMSPropOptimizer(learning_rate)#*max(1, hvd.size()-1))
    elif a.intr_opt == 'adm': opt = tf.train.AdamOptimizer(learning_rate)#*max(1, hvd.size()-1))
    elif a.intr_opt == 'sgd': opt = tf.train.GradientDescentOptimizer(learning_rate)

    # Horovod: add Horovod Distributed Optimizer.
    if a.dist_opt == 'fmb': dist = hvd.FixedMiniBatchDistributor(opt, a.batch_size)
    elif a.dist_opt == 'amb': dist = hvd.AnytimeMiniBatchDistributor(opt, a.batch_size,
                                                a.amb_time_limit, a.amb_num_splits)

    train_op = dist.minimize(placeholders, model_fac, global_step=global_step)
    accuracy, loss = model_fac.get_metrics()

    hooks = [hvd.BroadcastVariablesHook(dist.get_variables(), 0),
             tf.train.StopAtStepHook(last_step=a.last_step),]# // hvd.size()),
    if hvd.rank()==0:
        hooks.append(hvd.CSVLoggingHook(every_n_iter=a.log_freq,
                     train_tensors={'step':global_step, 'loss':loss, 'learning_rate':learning_rate},
                     test_tensors={'accuracy':accuracy}, get_test_fd=get_test_fd)) # 'accuracy':accuracy

    with tf.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op, feed_dict=get_train_fd())


if __name__ == '__main__':
    args = parse_args()
    print('[Arguments]', vars(args))
    main(args)
