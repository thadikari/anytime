import tensorflow as tf
import argparse
import json
import os

import utilities as ut
import utilities.file
import distributed as hvd
import models

tf.logging.set_verbosity(tf.logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model', choices=models.reg.keys())
    parser.add_argument('dist_opt', choices=['fmb', 'amb'])
    parser.add_argument('intr_opt', choices=['sgd', 'rms', 'adm', 'mom'])
    parser.add_argument('batch_size', type=int)

    parser.add_argument('--amb_time_limit', type=float)
    parser.add_argument('--amb_num_partitions', type=int)

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
    parser.add_argument('--test_size', help='size of the subset from test dataset', default=-1, type=int)
    parser.add_argument('--data_dir', default=ut.file.resolve_data_dir('distributed'), type=str)
    parser.add_argument('--master_cpu', action='store_true')
    args = parser.parse_args()

    if args.dist_opt=='amb':
        if args.amb_time_limit is None or args.amb_num_partitions is None:
            parser.error('Need to define both amb_time_limit and amb_time_splis.')
        if args.amb_num_partitions > args.batch_size:
            parser.error('Case not allowed: args.amb_num_partitions > args.batch_sizes')

    return args


def main():
    extra_line = '' if _a.extra is None else '__%s'%_a.extra
    amb_args = f'__{_a.amb_time_limit:g}_{_a.amb_num_partitions}' if _a.dist_opt=='amb' else ''

    # Horovod: initialize Horovod.
    hvd.init(induce_stragglers=_a.induce, induce_dist=_a.dist)
    num_workers = hvd.num_workers()
    run_id = f'{_a.model}{extra_line}__{_a.dist_opt}_{_a.intr_opt}_{_a.batch_size}{amb_args}__{_a.starter_learning_rate:g}_{_a.decay_steps}_{_a.decay_rate:g}__{_a.induce}_{num_workers}'
    print('run_id: %s'%run_id)
    logs_dir = None if _a.no_stats else os.path.join(_a.data_dir, run_id)

    if hvd.is_master():
        if _a.master_cpu: os.environ['CUDA_VISIBLE_DEVICES'] = ''
        if logs_dir is not None:
            if not os.path.exists(logs_dir): os.makedirs(logs_dir)
            with open(os.path.join(logs_dir, 'args.json'), 'w') as fp_:
                ddd = vars(_a)
                ddd['num_workers'] = num_workers
                json.dump(ddd, fp_, indent=4)
    hvd.set_work_dir(logs_dir)

    model = models.reg.get(_a.model)
    placeholders, create_model_get_sum_loss, (get_train_fd, get_test_fd), init_call = model(_a.batch_size, _a.test_size)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.compat.v1.train.exponential_decay(_a.starter_learning_rate,
                    global_step, _a.decay_steps, _a.decay_rate, staircase=True)

    # Horovod: adjust learning rate based on number of GPUs.
    # in anytime impl this adjustment is made internally, by the number of steps returned by each worker
    if _a.intr_opt == 'rms': opt = tf.train.RMSPropOptimizer(learning_rate)#*max(1, hvd.size()-1))
    elif _a.intr_opt == 'adm': opt = tf.train.AdamOptimizer(learning_rate)#*max(1, hvd.size()-1))
    elif _a.intr_opt == 'mom': opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    elif _a.intr_opt == 'sgd': opt = tf.train.GradientDescentOptimizer(learning_rate)

    # Horovod: add Horovod Distributed Optimizer.
    if _a.dist_opt == 'fmb': dist = hvd.FixedMiniBatchDistributor(opt)
    elif _a.dist_opt == 'amb': dist = hvd.AnytimeMiniBatchDistributor(opt,
                                      _a.amb_time_limit, _a.amb_num_partitions)

    train_op = dist.minimize(placeholders, create_model_get_sum_loss, global_step=global_step)
    accuracy, avg_loss = create_model_get_sum_loss.get_metrics()

    hooks = [hvd.BroadcastVariablesHook(dist),
             tf.train.StopAtStepHook(last_step=_a.last_step)]
    if hvd.is_master():
        hooks.append(hvd.CSVLoggingHook(every_n_iter=_a.log_freq,
                     train_tensors={'step':global_step, 'loss':avg_loss, 'learning_rate':learning_rate},
                     test_tensors={'accuracy':accuracy}, get_test_fd=get_test_fd)) # 'accuracy':accuracy

    with tf.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
        mon_sess.run_step_fn(lambda step_context: init_call(step_context.session))
        while not mon_sess.should_stop():
            mon_sess.run(train_op, feed_dict=get_train_fd())


if __name__ == '__main__':
    _a = parse_args()
    print('[Arguments]', vars(_a))
    main()
