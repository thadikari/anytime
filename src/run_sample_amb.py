import tensorflow as tf
import time


'''
Use `compute_gradients` instead of `tf.Optimizer.compute_gradients`
to enable AMB.
'''

def compute_gradients(time_limit, num_partitions, create_model,
                        placeholders, variable_shapes, optimizer,
                        model_metrics=None, batch_size=None):
    '''
    This function does the following:
    1. divide input placeholders into parts
    2. computes gradients for each partition
    3. keeps track of time and stops computation if time limit is reached
    4. accumilates the gradients for each partition

    Args:
        time_limit (float):
            AMB time limit in seconds. Computation will stop when `time_limit`
            is reached. However, will wait until at least one gradient is
            computed. Otherwise will return with an all zeros gradient.

        num_partitions (int):
            Number of parts minibatch must be divided into.
            The `batch_size` must be a multiple of `num_partitions`.
            If `batch_size` is 99 and `num_partitions` is 9, AMB will
            compute 99/9=11 samples at a time until timeout occurs.

        create_model (callable):
            This callable should create the model. It must return two arguments.
            The first should be the *sum* of the losses of all data points.
            The second must be a list of other optional metrics such as
            accuracy. The signature of optional arguments should be as defined
            in `metrics` argument below. It is important to return *sum* of the
            losses and not the *average*. The callable must accept a list of
            arguments that are of same signature as the ones in the
            `placeholders` argument below. This function is not intended to be
            called by the user. It will be callled within `compute_gradients`.

        placeholders (list):
            Input arguments to `create_model` such as images, labels. These can
            be placeholders or tensors produced by `tf.data.Dataset`. Each item
            in this list will be partitioned across the first dimension and
            passed onto the callable that creates the model. This list must
            contain at least one element.

        variable_shapes (list):
            List of shapes of the varibles that will be created by the model.
            Order of items must match with the variables created within the
            model. This list is required before creating the model due to a
            limitation of `tf.while_loop`. See the implementation below for
            more information. To get the list of shapes in a model one can
            first create the model and print to console the following list:
            `list(v_.shape.as_list() for v_ in tf.trainable_variables())`.

        optimizer (tf.Optimizer):
            The base optimizer that is used to apply gradients.

        model_metrics (list, optional):
            Signature of the optional return arguments of `create_model`.
            For example, if average loss and accuracy is returned pass
            `model_metrics = [0., 0.]`.

        batch_size (int, optional):
            The minibatch size. Should be large enough so that AMB will not be
            able to go through the whole minibatch before the time limit.
            Otherwise a warning will be printed to stdout in runtime:
            `Increase batch_size or decrease time_limit!`.

    Returns:
        list: `grads_and_vars` similar to `tf.Optimizer.compute_gradients`.
        int: Number of gradients computed by the time limit.
        list: Optional metrics returned by `create_model`. Has the same
            signature as defined in `model_metrics`.
    '''

    if batch_size is None: batch_size = tf.shape(placeholders[0])[0]
    partition_size = tf.dtypes.cast(batch_size/num_partitions, tf.int32)

    class Timer:
        def reset(self): self.last = time.time()
        def elapsed(self): return time.time() - self.last

    timer = Timer()

    def cond(curr_partition, accs, metrics):
        # chk0: Make sure to complete at least one partition, otherwise
        # will return with an all zeros gradient.
        chk0 = tf.equal(curr_partition, 0)

        # chk1: Check if have gone through all the partitions (before
        # reaching the time limit). If this happens increase the batch size.
        prnt = lambda: print('Increase batch_size or decrease time_limit!')
        warn = lambda: tf.py_func(func=prnt, inp=[], Tout=[])
        okay = lambda: tf.no_op()
        chck = lambda: timer.elapsed() < time_limit
        start_, end_ = start_end(curr_partition)
        chk1 = tf.shape(placeholders[0][start_:end_])[0] > 0

        # chk2: Check whether the time limit is reached.
        chk2 = tf.py_func(func=chck, inp=[], Tout=tf.bool)

        with tf.control_dependencies([tf.cond(chk1, okay, warn)]):
            return tf.math.logical_or(chk0, tf.math.logical_and(chk1, chk2))

    def start_end(curr_partition):
        start_ = curr_partition*partition_size
        end_ = start_ + partition_size
        return start_, end_

    def body(curr_partition, accs, metrics):
        start_, end_ = start_end(curr_partition)
        partitioned = (pl[start_:end_] for pl in placeholders)
        sum_loss, metrics = create_model(*partitioned)
        grads_and_vars = optimizer.compute_gradients(sum_loss)
        grads, body.vars = zip(*grads_and_vars)
        ret_accs = list(acc+grad for acc,grad in zip(accs, grads))
        with tf.control_dependencies(ret_accs):
            return curr_partition+1, ret_accs, list(metrics)

    reset_timer = tf.py_func(func=lambda:timer.reset(), inp=[], Tout=[])
    with tf.control_dependencies([reset_timer]):
        accs_0 = list(tf.zeros(shape) for shape in variable_shapes)
        if model_metrics is None: model_metrics = []
        loop_vars = [tf.constant(0), accs_0, model_metrics]
        num_completed_pars, grads, opt_metrics = tf.while_loop(cond, body,
                                            loop_vars, swap_memory=True,
                                            parallel_iterations=1,
                                            return_same_structure=True)
    num_computed = num_completed_pars*partition_size
    return list(zip(grads, body.vars)), num_computed, opt_metrics



'''
Example on how to use `compute_gradients`.
'''

variable_shapes = [[10,500], [500], [500,10], [10]]
model_metrics = [0., 0.]

def create_model(feature, target):
    layers = tf.layers
    target_1h = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)
    lay = layers.dense(feature, 500, activation=tf.nn.relu)
    logits = layers.dense(lay, 10, activation=None)
    losses = tf.losses.softmax_cross_entropy(onehot_labels=target_1h,
                                        logits=logits, reduction='none')
    is_eq = tf.equal(tf.argmax(logits, 1), tf.cast(target, 'int64'))
    accuracy = tf.reduce_mean(tf.cast(is_eq, tf.float32))
    optional_metrics = [tf.reduce_mean(losses), accuracy]
    return tf.reduce_sum(losses), optional_metrics


import numpy as np
def create_data_for_one_iteration(batch_size):
    # Data generator for a dummy two class problem.
    class_size = int(batch_size/2)
    return np.vstack((np.random.normal(loc=1, size=[class_size,10]),
                      np.random.normal(loc=9, size=[class_size,10]))),\
           np.hstack((np.zeros(class_size), np.ones(class_size)))


amb_time_limit = 0.0010
amb_num_partitions = 20
batch_size = 100000
learning_rate = 0.0001
tf.logging.set_verbosity(tf.logging.INFO)

x = tf.placeholder(tf.float32, shape=(None, 10), name='feature')
y =  tf.placeholder(tf.float32, shape=None, name='target')
opt = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars, _, (avg_loss, accuracy) = compute_gradients(
                                    amb_time_limit, amb_num_partitions,
                                    create_model, [x, y],
                                    variable_shapes, opt, model_metrics)
train_op = opt.apply_gradients(grads_and_vars)

with tf.train.MonitoredTrainingSession() as mon_sess:
    while not mon_sess.should_stop():
        x_, y_ = create_data_for_one_iteration(batch_size)
        eval_l = [train_op, avg_loss, accuracy]
        _, al_, ac_ = mon_sess.run(eval_l, feed_dict={x:x_, y:y_})
        print('Avg. loss: %g, Accuracy: %g'%(al_,ac_))
