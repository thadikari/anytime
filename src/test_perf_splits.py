import tensorflow as tf
import numpy as np
import argparse
import time
import json
import os

from models import mnist, cifar10, toy_model
import utils

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def safe_read_json(file_path):
    try: dd = json.load(open(file_path))
    except: dd = {}
    return dd

def safe_get_key(dd, key, val):
    if key not in dd: dd[key] = val
    return dd[key]

shapes = {'cifar10': [[3, 3, 3, 64], [3, 3, 64, 128], [5, 5, 128, 256], [5, 5, 256, 512], [64], [64], [128], [128], [256], [256], [512], [512], [2048, 128], [128], [128], [128], [128, 256], [256], [256], [256], [256, 512], [512], [512], [512], [512, 10], [10]],
          'mnist': [(5,5,1,32), (32), (5,5,32,64), (64), (3136,1024), (1024), (1024,10), (10)],
          'toy_model': toy_model.make_model.shapes}


# https://stackoverflow.com/questions/49555016/compute-gradients-for-each-time-step-of-tf-while-loop
# Compute gradient inside tf.while_loop using TensorArray >> https://github.com/tensorflow/tensorflow/issues/9450
# what parallel_iterations does >> https://github.com/tensorflow/tensorflow/issues/1984

def log_d(fmt, *args):
    op = tf.py_func(func=print, inp=[fmt]+[*args], Tout=[])
    return tf.control_dependencies([op])

def main():
    batch_size, num_splits = args.batch_size, args.num_splits
    split_size = int(batch_size/num_splits)

    model = globals()[args.model]
    placeholders, model_fac, get_train_fd, get_test_fd = model.get_fac_elements(batch_size)
    features_pl, labels_pl = placeholders
    opt = tf.train.AdamOptimizer(0.0001)

    global vars
    vars = None

    def cond(curr_split, _, *accs):
        return curr_split < num_splits

    def get_grads(features, labels):
        global vars
        loss = model_fac(features, labels)
        gradients = opt.compute_gradients(loss)
        grads, vars = zip(*gradients)
        return loss, grads

    def body(curr_split, _, *accs):
        start_ = curr_split*split_size
        end_ = start_ + split_size
        loss, grads = get_grads(features_pl[start_:end_], labels_pl[start_:end_])
        ret_accs = list(acc+grad for acc,grad in zip(accs, grads))
        # a0, g0, r0 = accs[0], grads[0], ret_accs[0]
        # log_op = tf.py_func(func=log_, inp=[loss, curr_split], Tout=[])
        with tf.control_dependencies(list(grads)):
            return [curr_split+1, loss] + ret_accs

    if num_splits==1:
        loss, grads = get_grads(features_pl, labels_pl)
    else:
        accs_0 = list(tf.zeros(shape) for shape in shapes[args.model])
        _, loss, *grads = tf.while_loop(cond, body, [tf.constant(0), 0.] + accs_0,
                         parallel_iterations=1, return_same_structure=True, swap_memory=True)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    grad_vars = zip(grads, vars)
    eval_op = opt.apply_gradients(grad_vars, global_step=global_step)

    hooks = [tf.estimator.LoggingTensorHook(tensors={'loss':loss}, every_n_iter=args.log_freq),
             tf.train.StopAtStepHook(last_step=args.last_step)]
    with tf.compat.v1.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
        elapsed = 0.
        while not mon_sess.should_stop():
            train_fd = get_train_fd()
            start_t = time.time()
            _, loss_ = mon_sess.run([eval_op, loss], feed_dict=train_fd)
            elapsed += time.time()-start_t


    file_path = os.path.join(args.data_dir, 'splits.json')
    data = safe_read_json(file_path)

    dd = safe_get_key(safe_get_key(data, str(args.batch_size), {}), str(args.num_splits), {})
    dd['time_per_step'] = elapsed*1.0/args.last_step
    dd['last_loss'] = loss_.item()

    print(data)
    json.dump(data, open(file_path, 'w'), sort_keys=True, indent=4)


def run_batch():
    import subprocess
    for i in range(2,20):
        for j in range(i):
            print(2**i,2**j)
            subprocess.call(['python', '-u', 'test_slices.py', 'main',
                             '--model', args.model,
                             '--batch_size', str(2**i),
                             '--num_splits', str(2**j)])

def plot():
    import matplotlib.pyplot as plt
    from pathlib import Path

    file_name = 'ec2-m3-xlarge.json'
    file_name = 'splits.json'
    file_path = os.path.join('..', 'data', 'test_slices', file_name)
    data = safe_read_json(file_path)
    ax1, ax2 = plt.subplot(121), plt.subplot(122)
    for batch_size in sorted(map(int, list(data.keys()))):
        dd = data[str(batch_size)]
        num_splits_ll = np.array(list(sorted(map(int, list(dd.keys())))))
        time_per_step_ll = np.array(list(dd[str(ns)]['time_per_step'] for ns in num_splits_ll))
        time_per_sample_ll = time_per_step_ll/batch_size
        micro_batch_size_ll = batch_size/num_splits_ll
        # ax.plot(num_splits_ll, time_per_step_ll, label='Batch size=%d'%batch_size)
        ax1.plot(num_splits_ll, time_per_step_ll, label='Batch size=%d'%batch_size)
        ax2.plot(micro_batch_size_ll, time_per_sample_ll, label='Batch size=%d'%batch_size)

    ax1.set_xlabel('Number of splits'); ax1.set_ylabel('Time per step')
    ax1.set_xscale('log', basex=2); ax1.set_yscale('log'); ax1.legend(loc='best')

    ax2.set_xlabel('Split size'); ax2.set_ylabel('Time per sample')
    ax2.set_xscale('log', basex=2); ax2.set_yscale('log'); ax2.legend(loc='best')

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['batch', 'main', 'plot'])
    parser.add_argument('--model', choices=['mnist', 'cifar10', 'toy_model'])
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_splits', type=int)
    parser.add_argument('--last_step', type=int, default=10)
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--data_dir', default=utils.resolve_data_dir('distributed'))
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print('[Arguments]', vars(args))
    if args.type=='batch': run_batch()
    elif args.type=='main': main()
    elif args.type=='plot': plot()
