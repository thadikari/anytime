from tensorflow import keras
import tensorflow as tf
import numpy as np
import os


layers = tf.layers

def conv_model(feature, target):
    """2-layer convolution model."""
    # Convert the target to a one-hot tensor of shape (batch_size, 10) and
    # with a on-value of 1 for each one-hot vector of length 10.
    target_1h = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)

    # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
    # image width and height final dimension being the number of color channels.
    feature = tf.reshape(feature, [-1, 28, 28, 1])

    # First conv layer will compute 32 features for each 5x5 patch
    with tf.compat.v1.variable_scope('conv_layer1'):
        h_conv1 = layers.conv2d(feature, 32, kernel_size=[5, 5],
                                activation=tf.nn.relu, padding="SAME")
        h_pool1 = tf.nn.max_pool2d(
            h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second conv layer will compute 64 features for each 5x5 patch.
    with tf.compat.v1.variable_scope('conv_layer2'):
        h_conv2 = layers.conv2d(h_pool1, 64, kernel_size=[5, 5],
                                activation=tf.nn.relu, padding="SAME")
        h_pool2 = tf.nn.max_pool2d(
            h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # Densely connected layer with 1024 neurons.
    h_fc1 = layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu)#layers.dropout(#,
        #rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Compute logits (1 per class) and compute loss.
    logits = layers.dense(h_fc1, 10, activation=None)
    # loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(target_1h, logits, reduction='none'))
    loss = tf.compat.v1.losses.softmax_cross_entropy(target_1h, logits)
    predict = tf.argmax(logits, 1)
    is_correct = tf.equal(tf.cast(target, 'int64'), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, 'float'))
    # error = 100*(1 - tf.reduce_mean(tf.cast(self.is_correct, 'float')))
    return accuracy, loss

def get_mnist():
    # Keras automatically creates a cache directory in ~/.keras/datasets for
    # storing the downloaded MNIST data. This creates a race
    # condition among the workers that share the same filesystem. If the
    # directory already exists by the time this worker gets around to creating
    # it, ignore the resulting exception and continue.
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
    if not os.path.exists(cache_dir):
        try:
            os.mkdir(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
                pass
            else:
                raise

    (x_train_, y_train), (x_test_, y_test) = keras.datasets.mnist.load_data('MNIST-data')
    x_train = np.reshape(x_train_, (-1, 784)) / 255.0
    x_test = np.reshape(x_test_, (-1, 784)) / 255.0
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    return (x_train, y_train), (x_test, y_test)

def permute(x_, y_, seed=None):
    p = np.random.RandomState(seed=seed).permutation(len(x_))
    return x_[p], y_[p]


def input_generator(x_train, y_train, batch_size):
    while True:
        x_train, y_train = permute(x_train, y_train)
        index = 0
        while index <= len(x_train) - batch_size:
            yield x_train[index:index + batch_size], \
                  y_train[index:index + batch_size],
            index += batch_size


def make_input_generator(batch_size):
    (x_train, y_train), (x_test, y_test) = get_mnist()
    return input_generator(x_train, y_train, batch_size)


def get_everything(batch_size, test_size=100):
    image, label = tf.placeholder(tf.float32, [None, 784], name='image'), tf.placeholder(tf.float32, [None], name='label')
    accuracy, loss = conv_model(image, label)
    (x_train, y_train), (x_test, y_test) = get_mnist()
    generator = input_generator(x_train, y_train, batch_size)
    x_test, y_test = permute(x_test, y_test, seed=test_size)
    x_test, y_test = x_test[:test_size], y_test[:test_size]
    def get_train_fd():
        return dict(zip([image, label], next(generator)))
    return loss, accuracy, get_train_fd, lambda: {image:x_test, label:y_test}
