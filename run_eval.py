import tensorflow as tf
import numpy as np
import os
import distributed as hvd

import cifar10
import mnist

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    SCRATCH = '/scratch/s/sdraper/tharindu'

    model = cifar10 # cifar10 mnist
    intr_opt = 'adm' # sgd rms adm
    dist_opt = 'fmb' # any fmb
    log_freq = 1
    starter_learning_rate = 0.001
    decay_steps, decay_rate = 10, 1.0
    batch_size = 64 # 1024 16 = 16384
    straggler_std_dev = 10
    fmb_every_n_batches = 16
    any_time_limit = 9.3
    last_step = 1000000

    run_id = f'{model.__name__}_{dist_opt}_{intr_opt}_{starter_learning_rate:g}_{decay_steps}_{decay_rate:g}_{batch_size}_{straggler_std_dev:g}_{fmb_every_n_batches}_{any_time_limit:g}'
    # print('run_id: %s'%run_id)
    logs_dir = os.path.join(SCRATCH, 'checkpoints', run_id)
    tmp_dir = os.path.join(SCRATCH, '.tmp')

    # Horovod: initialize Horovod.
    hvd.init(work_dir=logs_dir, straggler_std_dev_ms=straggler_std_dev)

    loss, get_train_fd = model.get_everything(batch_size, tmp_dir)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate,
                    global_step, decay_steps, decay_rate, staircase=True)

    # Horovod: adjust learning rate based on number of GPUs.
    # in anytime impl this adjustment is made internally, by the number of steps returned by each worker
    if intr_opt == 'rms':
        opt = tf.train.RMSPropOptimizer(learning_rate)#*max(1, hvd.size()-1))
    elif intr_opt == 'adm':
        opt = tf.train.AdamOptimizer(learning_rate)#*max(1, hvd.size()-1))
    elif intr_opt == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    else: raise Exception('intr_opt error')

    # Horovod: add Horovod Distributed Optimizer.
    if dist_opt == 'fmb':
        opt = hvd.FixedMiniBatchOptimizer(opt, every_n_batches=fmb_every_n_batches)
    elif dist_opt == 'any':
        opt = hvd.AnytimeOptimizer(opt, time_limit_sec=any_time_limit)
    else: raise Exception('dist_opt error')

    train_op = opt.minimize(loss, global_step=global_step)

    hooks = [hvd.BroadcastGlobalVariablesHook(0),
             tf.train.StopAtStepHook(last_step=last_step),]# // hvd.size()),
    if hvd.rank()==0:
        for hook in [tf.train.LoggingTensorHook, hvd.CSVLoggingHook]:
            hooks.append(hook(every_n_iter=log_freq, 
                              tensors={'step': global_step, 
                                       'loss': loss,
                                       'learning_rate': learning_rate}))

    with tf.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op, feed_dict=get_train_fd())

if __name__ == '__main__':
    tf.app.run()
