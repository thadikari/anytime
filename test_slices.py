import tensorflow as tf
import numpy as np
import argparse
import time
import json
import os

import mnist

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def safe_read_json(file_path):
    try: dd = json.load(open(file_path))
    except: dd = {}
    return dd

def safe_get_key(dd, key, val):
    if key not in dd: dd[key] = val
    return dd[key]


# https://stackoverflow.com/questions/49555016/compute-gradients-for-each-time-step-of-tf-while-loop
# Compute gradient inside tf.while_loop using TensorArray >> https://github.com/tensorflow/tensorflow/issues/9450
# what parallel_iterations does >> https://github.com/tensorflow/tensorflow/issues/1984

def main():
    features_pl = tf.compat.v1.placeholder(tf.float32, [None, 784], name='image')
    labels_pl = tf.compat.v1.placeholder(tf.float32, [None], name='label')
    opt = tf.train.AdamOptimizer(0.0001)

    # batch_size, num_splits = 2**8, 1
    log_every_n_iter = 1
    batch_size, num_splits = args.batch_size, args.num_splits
    split_size = int(batch_size/num_splits)
    global vars
    vars = None

    def log_(loss, curr_split):#, a0, g0, r0):
        print('loss: %g, curr_split: %s'%(loss, curr_split))
        # , a0[0][0][0][0], g0[0][0][0][0], r0[0][0][0][0]))

    def cond_func():
        print(time.time())

    def cond(curr_split, _, *accs):
        log_func = tf.py_func(func=cond_func, inp=[], Tout=[])
        #with tf.control_dependencies([log_func]):
        return curr_split < num_splits

    def get_grads(features, labels):
        global vars
        _, loss = mnist.conv_model(features, labels)
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

    def create_accs():
        shapes = [(5,5,1,32), (32), (5,5,32,64), (64),
                  (3136,1024), (1024), (1024,10), (10)]
        return list(tf.zeros(shape) for shape in shapes)

    if num_splits==1:
        loss, grads = get_grads(features_pl, labels_pl)
    else:
        _, loss, *grads = tf.while_loop(cond, body, [tf.constant(0), 0.] + create_accs(),
                         parallel_iterations=1, return_same_structure=True,
                         swap_memory=True)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    grad_vars = zip(grads, vars)
    eval_op = opt.apply_gradients(grad_vars, global_step=global_step)

    hooks = [tf.estimator.LoggingTensorHook(tensors={'loss':loss}, every_n_iter=log_every_n_iter),
             tf.train.StopAtStepHook(last_step=args.last_step)]
    start_t = time.time()
    with tf.compat.v1.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
        generator = mnist.make_input_generator(batch_size=batch_size)
        while not mon_sess.should_stop():
            _, loss_ = mon_sess.run([eval_op, loss], feed_dict=dict(zip([features_pl, labels_pl], next(generator))))

    elapsed = time.time()-start_t

    file_path = 'data.json'
    data = safe_read_json(file_path)

    dd = safe_get_key(safe_get_key(data, str(args.batch_size), {}), str(args.num_splits), {})
    dd['time_per_step'] = elapsed*1.0/args.last_step
    dd['last_loss'] = loss_.item()
    
    print(data)
    json.dump(data, open(file_path, 'w'), sort_keys=True, indent=4)


def run_batch():
    import subprocess
    for i in range(12,14):
        for j in range(i):
            print(2**i,2**j)
            subprocess.call(['python', '-u', 'test_slices.py', 'main', 
                             '--batch_size', str(2**i),
                             '--num_splits', str(2**j)])

def plot():
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    file_path = 'data.json'
    data = safe_read_json(file_path)
    ax = plt.gca()
    for batch_size in sorted(map(int, list(data.keys()))):
        dd = data[str(batch_size)]
        num_splits_ll = np.array(list(sorted(map(int, list(dd.keys())))))
        time_per_step_ll = np.array(list(dd[str(ns)]['time_per_step'] for ns in num_splits_ll))
        time_per_sample_ll = time_per_step_ll/batch_size
        micro_batch_size_ll = batch_size/num_splits_ll
        # ax.plot(num_splits_ll, time_per_step_ll, label='Batch size=%d'%batch_size)
        ax.plot(micro_batch_size_ll, time_per_sample_ll, label='Batch size=%d'%batch_size)

    # ax.set_xlabel('Number of splits'); ax.set_ylabel('Time per step')
    ax.set_xlabel('Micro batch size'); ax.set_ylabel('Time per sample')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.legend(loc='best')
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['batch', 'main', 'plot'])
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_splits', type=int)
    parser.add_argument('--last_step', type=int, default=10)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print('[Arguments]', vars(args))
    if args.type=='batch': run_batch()
    elif args.type=='main': main()
    elif args.type=='plot': plot()
