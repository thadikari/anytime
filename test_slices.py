import tensorflow as tf
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def get_mnist():
    from tensorflow import keras
    (x_train_, y_train), *rest = keras.datasets.mnist.load_data('MNIST-data')
    x_train = np.reshape(x_train_, (-1, 784)) / 255.0
    assert len(x_train) == len(y_train)
    return (x_train, y_train)


def train_input_generator(batch_size):
    features, labels = get_mnist()
    while True:
        p = np.random.permutation(len(features))
        features, labels = features[p], labels[p]
        index = 0
        while index <= len(features) - batch_size:
            yield features[index:index + batch_size], \
                  labels[index:index + batch_size],
            index += batch_size


def conv_model(feature, target):
    feature = tf.reshape(feature, [-1, 28, 28, 1])
    target_1h = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)
    # h_pool2_flat = tf.reshape(feature, [-1, 784])

    with tf.variable_scope('conv_layer1'):
        h_conv1 = tf.layers.conv2d(feature, 32, kernel_size=[5, 5],
                                activation=tf.nn.relu, padding="SAME")
        h_pool1 = tf.nn.max_pool(
            h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv_layer2'):
        h_conv2 = tf.layers.conv2d(h_pool1, 64, kernel_size=[5, 5],
                                activation=tf.nn.relu, padding="SAME")
        h_pool2 = tf.nn.max_pool(
            h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    h_fc1 = tf.layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu)
    logits = tf.layers.dense(h_fc1, 10, activation=None)
    return tf.losses.softmax_cross_entropy(target_1h, logits)


# https://stackoverflow.com/questions/49555016/compute-gradients-for-each-time-step-of-tf-while-loop
# Compute gradient inside tf.while_loop using TensorArray >> https://github.com/tensorflow/tensorflow/issues/9450
# what parallel_iterations does >> https://github.com/tensorflow/tensorflow/issues/1984

def main():
    features_pl = tf.compat.v1.placeholder(tf.float32, [None, 784], name='image')
    labels_pl = tf.compat.v1.placeholder(tf.float32, [None], name='label')

    batch_size, num_splits, log_every_n_iter = 60000, 1, 1
    split_size = int(batch_size/num_splits)
    
    def log_(loss, curr_split):#, a0, g0, r0):
        print('loss: %g, curr_split: %s'%(loss, curr_split))
        # , a0[0][0][0][0], g0[0][0][0][0], r0[0][0][0][0]))

    def cond(curr_split, *accs):
        return curr_split < num_splits

    def body(curr_split, *accs):
        start_ = curr_split*split_size
        end_ = start_ + split_size
        loss = conv_model(features_pl[start_:end_], labels_pl[start_:end_])
        gradients = tf.train.Optimizer(False, 'opt').compute_gradients(loss)
        grads, _ = zip(*gradients)
        ret_accs = list(acc+grad for acc,grad in zip(accs, grads))
        # a0, g0, r0 = accs[0], grads[0], ret_accs[0]
        log_op = tf.py_func(func=log_, inp=[loss, curr_split], Tout=[])
        with tf.control_dependencies(list(grads)):
            return [curr_split+1] + ret_accs

    def create_accs():
        shapes = [(5,5,1,32), (32), (5,5,32,64), (64), 
                  (3136,1024), (1024), (1024,10), (10)]
        return list(tf.zeros(shape) for shape in shapes)
    
    _, *accs = tf.while_loop(cond, body, [tf.constant(0)] + create_accs(), 
                         parallel_iterations=1, return_same_structure=True, 
                         swap_memory=True)
    eval_op = accs

    hooks = [tf.estimator.LoggingTensorHook(tensors={}, every_n_iter=log_every_n_iter)]
    with tf.compat.v1.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
        generator = train_input_generator(batch_size=batch_size)
        while not mon_sess.should_stop():
            mon_sess.run(eval_op, feed_dict=dict(zip([features_pl, labels_pl], next(generator))))

main()