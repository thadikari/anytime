import tensorflow as tf
import argparse
import json
import os

import utilities as ut
import utilities.file
import utilities.learningrate
import strategy as sgy
import nodes as nds
import models

tf.logging.set_verbosity(tf.logging.INFO)


opt_reg = utilities.Registry()
_r = opt_reg.put
_r('rms', tf.train.RMSPropOptimizer)
_r('adm', tf.train.AdamOptimizer)
_r('sgd', tf.train.GradientDescentOptimizer)
_r('mom', lambda lr_: tf.train.MomentumOptimizer(lr_, momentum=0.9))

sgy_reg = {'syncr':sgy.SynchronousFac, 'async':sgy.AsynchronousFac}

straggler_reg = utilities.Registry()
_s = straggler_reg.put
_s('niagara', [[0, 0, .4], [0, .4, .3], [2, .3, .2], [5, .3, .1]])
_s('cedar', [[0, 0, .4], [.05, .04, .3], [0.15, .03, .2], [0.25, .03, .1]])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model', choices=models.reg.keys())
    parser.add_argument('dist_opt', choices=['fmb', 'amb'])
    parser.add_argument('intr_opt', choices=opt_reg.keys())
    parser.add_argument('batch_size', type=int)

    parser.add_argument('--amb_time_limit', type=float)
    parser.add_argument('--amb_num_partitions', type=int)

    parser.add_argument('--dist_sgy', help='distributed consensus strategy', choices=sgy_reg.keys(), default='syncr')
    parser.add_argument('--async_master', help='async master master waiting style', choices=['time', 'batch'], default='batch')
    parser.add_argument('--async_master_time_limit', type=float, default=0.1)
    parser.add_argument('--async_master_batch_min', type=int, default=4)
    parser.add_argument('--async_delay_std', help='std.dev of 0 mean gaussian, delay=abs(sample)', type=float, default=-1)
    parser.add_argument('--induce_comp', help='distribution for inducing computation stragglers', choices=straggler_reg.keys())

    parser.add_argument('--log_freq', default=1, type=int)
    parser.add_argument('--last_step', default=400000, type=int)
    parser.add_argument('--test_size', help='size of the subset from test dataset', default=-1, type=int)
    parser.add_argument('--data_dir', default=ut.file.resolve_data_dir('distributed'), type=str)

    parser.add_argument('--cuda', help='set CUDA_VISIBLE_DEVICES differently if on same node', choices=['cpu_master', 'gpu_master', 'gpu_all'])

    ut.learningrate.bind_learning_rates(parser)
    args = parser.parse_args()

    if args.dist_opt=='amb':
        if args.amb_time_limit is None or args.amb_num_partitions is None:
            parser.error('Need to define both amb_time_limit and amb_time_splis.')
        if args.amb_num_partitions > args.batch_size:
            parser.error('Case not allowed: args.amb_num_partitions > args.batch_sizes')

    return args


def main():
    amb_args = f'__{_a.amb_time_limit:g}_{_a.amb_num_partitions}' if _a.dist_opt=='amb' else ''
    sgy_args = f'__{_a.dist_sgy}'
    if _a.dist_sgy=='async':
        if _a.async_master=='batch': ext_str = f'{_a.async_master_batch_min}'
        elif _a.async_master=='time': ext_str = f'{_a.async_master_time_limit:g}'
        sgy_args = f'{sgy_args}_{_a.async_master}_{ext_str}_{_a.async_delay_std:g}'

    sgy.init() # log_level=(not _a.no_stats))
    num_workers = sgy.num_workers()
    scheduler = ut.learningrate.lrate_scheduler(_a)(_a, _a.last_step)
    lrate_str = scheduler.to_str()
    run_id = f'{_a.model}__{_a.dist_opt}_{_a.intr_opt}_{_a.batch_size}{amb_args}{sgy_args}__{lrate_str}__{_a.induce_comp}_{num_workers}'
    logs_dir = os.path.join(_a.data_dir, run_id)
    ds_args = {'batch_size':_a.batch_size, 'test_size':_a.test_size}

    if (sgy.is_master() and _a.cuda=='cpu_master') or (not sgy.is_master() and _a.cuda=='gpu_master'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif _a.cuda=='gpu_all':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(sgy.rank())
    print('CUDA_VISIBLE_DEVICES [rank %d]: %s'%(sgy.rank(), str(os.environ.get('CUDA_VISIBLE_DEVICES', None))))

    if sgy.is_master():
        print('run_id: %s'%run_id)
        if logs_dir is not None:
            if not os.path.exists(logs_dir): os.makedirs(logs_dir)
            print('logs dir: %s'%logs_dir)
            with open(os.path.join(logs_dir, 'args.json'), 'w') as fp_:
                ddd = vars(_a)
                ddd['num_workers'] = num_workers
                ddd['induce_comp_dist'] = straggler_reg.get(_a.induce_comp, None)
                json.dump(ddd, fp_, indent=4)
    else:
        ds_args.update({'num_workers':num_workers, 'worker_index':sgy.rank()-1})

    model = models.reg.get(_a.model)
    placeholders, create_model_get_sum_loss, (get_train_fd, get_test_fd), init_call = model(ds_args)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = opt_reg.get(_a.intr_opt)(learning_rate)

    if _a.dist_opt == 'fmb': dist = nds.FixedMiniBatchDistributor(opt)
    elif _a.dist_opt == 'amb': dist = nds.AnytimeMiniBatchDistributor(opt,
                                      _a.amb_time_limit, _a.amb_num_partitions)
    if _a.induce_comp is not None: dist.set_straggler(induce_dist=straggler_reg.get(_a.induce_comp))

    dist_sgy = sgy_reg[_a.dist_sgy](work_dir=logs_dir)
    if _a.dist_sgy=='async': dist_sgy.master_args(style=_a.async_master,
        batch_min=_a.async_master_batch_min, time_limit=_a.async_master_time_limit).set_straggler(delay_std=_a.async_delay_std)
    dist.set_strategy(dist_sgy)

    train_op = dist.minimize(placeholders, create_model_get_sum_loss, global_step=global_step)
    accuracy, avg_loss = create_model_get_sum_loss.get_metrics()

    hooks = [nds.BroadcastVariablesHook(dist),
             tf.train.StopAtStepHook(last_step=_a.last_step)]
    if sgy.is_master():
        hooks.append(nds.CSVLoggingHook(every_n_iter=_a.log_freq,
                     train_tensors={'step':global_step, 'loss':avg_loss, 'learning_rate':learning_rate},
                     test_tensors={'test_loss':avg_loss, 'accuracy':accuracy}, get_test_fd=get_test_fd,
                     work_dir=logs_dir)) # 'accuracy':accuracy

    with tf.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
        mon_sess.run_step_fn(lambda step_context: init_call(step_context.session))
        step = 0
        while not mon_sess.should_stop():
            lrate = scheduler(step)
            mon_sess.run(train_op, feed_dict={**get_train_fd(), learning_rate:lrate})
            step += 1


if __name__ == '__main__':
    _a = parse_args()
    print('[Arguments]', vars(_a))
    main()
