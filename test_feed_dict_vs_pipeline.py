#
# # Test conclusions:
# ++ input_method 1 (TF pipeline) is ~10x slower than 0 (feed_dict).
# ++ this is possibly due to the TF data conversion overhead described in following.

# https://stackoverflow.com/questions/51541610/why-is-tensorflows-tf-data-package-slowing-down-my-code
# https://www.tensorflow.org/guide/performance/datasets
# https://github.com/tensorflow/tensorflow/issues/15694


import tensorflow as tf
import numpy as np
from tensorflow import keras


tf.logging.set_verbosity(tf.logging.INFO)

def get_mnist():
    (x_train_, y_train), *rest = keras.datasets.mnist.load_data('MNIST-data')
    x_train = np.reshape(x_train_, (-1, 784)) / 255.0
    assert len(x_train) == len(y_train)
    return (x_train, y_train)

def train_input_generator(batch_size):
    features_placeholder = tf.placeholder(tf.float32, [None, 784], name='image')
    labels_placeholder = tf.placeholder(tf.float32, [None], name='label')

    def gen(features, labels):
        while True:
            p = np.random.permutation(len(features))
            features, labels = features[p], labels[p]
            index = 0
            while index <= len(features) - batch_size:
                yield features[index:index + batch_size], \
                      labels[index:index + batch_size],
                index += batch_size

    return [features_placeholder, labels_placeholder], gen(*get_mnist())

def train_input_pipeline(batch_size):
    features, labels = get_mnist()
    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    dataset = dataset.batch(batch_size).cache().repeat().prefetch(1)
    iterator = dataset.make_initializable_iterator()

    # https://stackoverflow.com/questions/45945881/tf-train-monitoredtrainingsession-and-reinitializable-iterator-from-dataset
    # https://github.com/tensorflow/tensorflow/issues/12859#issuecomment-371440926
    class _DatasetInitializerHook(tf.train.SessionRunHook):
        def begin(self): pass
        def after_create_session(self, sess, coord):
            sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                                      labels_placeholder: labels})
    return iterator.get_next(), _DatasetInitializerHook()


def conv_model(feature, target):
    target_1h = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)
    feature = tf.reshape(feature, [-1, 28, 28, 1])
    h_pool2_flat = tf.reshape(feature, [-1, 784])

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


def run0(loss, hooks):
    def sample(*inp): pass
    gradients = tf.train.Optimizer(False, 'opt').compute_gradients(loss)
    grads, vars = zip(*gradients)
    return tf.py_func(func=sample, inp=grads, Tout=[])
    # with tf.control_dependencies(grads): return tf.no_op()

def run1(loss, hooks):
    gradients = tf.train.Optimizer(False, 'opt').compute_gradients(loss)
    train_op = tf.group(*[var.assign_add(-0.01*grad) for grad, var in gradients])
    return train_op

def run2(loss, hooks):
    return tf.train.RMSPropOptimizer(0.001).minimize(loss)

def run3(loss, hooks):
    import distributed as hvd
    hvd.init()
    opt = tf.train.RMSPropOptimizer(0.001)
    opt = hvd.FixedMiniBatchOptimizer(opt, every_n_batches=fmb_every_n_batches)
    hooks.append(hvd.BroadcastGlobalVariablesHook(0))
    return opt.minimize(loss)

def main(run_def):
    if input_method==0:
        placeholders, generator = train_input_generator(batch_size=batch_size)
        init_hook = tf.train.SessionRunHook()
        def get_feed_dict(): return dict(zip(placeholders, next(generator)))
    else:
        placeholders, init_hook = train_input_pipeline(batch_size=batch_size)
        def get_feed_dict(): return None

    loss = conv_model(*placeholders)
    #global_step = tf.train.get_or_create_global_step()
    hooks = [init_hook,
             tf.train.LoggingTensorHook(tensors={'loss': loss},
                                        every_n_iter=log_every_n_iter)]
    train_op = run_def(loss, hooks)
    with tf.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op, feed_dict=get_feed_dict())


batch_size, log_every_n_iter = 64*1,1
batch_size, log_every_n_iter = 64,1
input_method = 0
main(run2)
