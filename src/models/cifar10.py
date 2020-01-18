import os
import random
import tarfile
import pickle
import numpy as np
import tensorflow as tf
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm


# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/cifar10.py
# https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428

cifar10_dataset_folder_name = 'cifar-10-batches-py'
data_directory = os.path.join(os.path.expanduser('~'), '.keras', 'cifar10')
tp_ = lambda *arg_: os.path.join(data_directory, *arg_)
log = lambda arg: print(arg)

class DownloadProgress(tqdm):
    last_block = 0
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def display_stats(batch_id, sample_id):
    features, labels = load_raw_batch('data_batch_%d'%batch_id)
    if not (0 <= sample_id < len(features)):
        log('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    log('\nStats of batch #{}:'.format(batch_id))
    log('# of Samples: {}\n'.format(len(features)))

    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    for key, value in label_counts.items():
        log('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]

    log('\nExample of Image {}:'.format(sample_id))
    log('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    log('Image - Shape: {}'.format(sample_image.shape))
    log('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))

def normalize(x):
    min_val, max_val = np.min(x), np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def one_hot_encode(x):
    encoded = np.zeros((len(x), 10))
    for idx, val in enumerate(x): encoded[idx][val] = 1
    return encoded

def load_raw_batch(file_name):
    with open(tp_(cifar10_dataset_folder_name, file_name), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1') # note the encoding type is 'latin1'
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels

def load_preproc_batch(file_name):
    file_path = tp_('preprocessed_%s'%file_name)
    if not isfile(file_path):
        log('Data file not available: {}'.format(file_path))
        get_data()
    return pickle.load(open(file_path, mode='rb'))

def do_all_(file_name):
    features, labels = load_raw_batch(file_name)
    features, labels = normalize(features), one_hot_encode(labels)
    pickle.dump((features, labels), open(tp_('preprocessed_%s'%file_name), 'wb'))

def get_data():
    # Download the dataset (if not exist yet)
    tar_path = tp_('cifar-10-python.tar.gz')
    if not isfile(tar_path):
        log('CIFAR-10 tar not available: {}'.format(tar_path))
        if not isdir(data_directory):
            log('Creating data directory: {}'.format(data_directory))
            os.makedirs(data_directory)
        log('Downloading data to: {}'.format(tar_path))
        with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
            urlretrieve(
                'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                tar_path,
                pbar.hook)

    if 1: #not isdir(tp_(cifar10_dataset_folder_name)):
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=tp_(''))
            tar.close()

    # Explore the dataset
    batch_id = 3
    sample_id = 7000
    display_stats(batch_id, sample_id)

    for batch_id in range(1,6): do_all_('data_batch_%d'%batch_id)
    do_all_('test_batch')

def conv_net(x, keep_prob):
    def initializer(shape): return (lambda: tf.truncated_normal(shape=shape, mean=0, stddev=0.08))
    conv1_filter = tf.Variable(initializer([3, 3, 3, 64]))
    conv2_filter = tf.Variable(initializer([3, 3, 64, 128]))
    conv3_filter = tf.Variable(initializer([5, 5, 128, 256]))
    conv4_filter = tf.Variable(initializer([5, 5, 256, 512]))

    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 3, 4
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    # 5, 6
    conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv3_bn = tf.layers.batch_normalization(conv3_pool)

    # 7, 8
    conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv4_bn = tf.layers.batch_normalization(conv4_pool)

    # 9
    flat = tf.contrib.layers.flatten(conv4_bn)

    # 10
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)

    # 11
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)

    # 12
    full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
    full3 = tf.nn.dropout(full3, keep_prob)
    full3 = tf.layers.batch_normalization(full3)

    # # 13
    # full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
    # full4 = tf.nn.dropout(full4, keep_prob)
    # full4 = tf.layers.batch_normalization(full4)

    return tf.contrib.layers.fully_connected(inputs=full3, num_outputs=10, activation_fn=None)

def make_model(x,y,keep_prob):
    logits = conv_net(x, keep_prob)
    losses = tf.losses.softmax_cross_entropy(y, logits, reduction='none')
    sum_loss = tf.reduce_sum(losses)
    avg_loss = tf.reduce_mean(losses)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    return accuracy, sum_loss, avg_loss

def permute(x_, y_, seed=None):
    p = np.random.RandomState(seed=seed).permutation(len(x_))
    return x_[p], y_[p]

def input_generator(batch_size):
    epoch = 0
    while True:
        n_batches = 5
        for batch_id in random.sample(range(1, n_batches + 1), n_batches):
            features, labels = permute(*load_preproc_batch('data_batch_%d'%batch_id))
            for start in range(0, len(features), batch_size):
                end = start+batch_size
                if end>len(features): continue
                yield features[start:end], labels[start:end]
            log('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_id))
        epoch += 1

# https://github.com/deep-diver/CIFAR10-img-classification-tensorflow
def get_fac_elements(batch_size, test_size=-1):

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
    y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    class ModelFac:
        def __call__(self, feature, target):
            self.accuracy, sum_loss, self.avg_loss = make_model(feature, target, keep_prob)
            return sum_loss

        def get_metrics(self):
            return self.accuracy, self.avg_loss

    # accuracy, loss = make_model(x,y,keep_prob)
    generator = input_generator(batch_size)

    x_test, y_test = load_preproc_batch('test_batch')
    if test_size>=0:
        x_test, y_test = permute(x_test, y_test, seed=test_size)
        x_test, y_test = x_test[:test_size], y_test[:test_size]

    def get_train_fd():
        x_, y_ = next(generator)
        return {x:x_, y:y_, keep_prob:0.7}
    return (x,y), ModelFac(), get_train_fd, lambda: {x:x_test, y:y_test, keep_prob:1.0}

#     valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
# accuracy
# loss = sess.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1. })
# valid_acc = sess.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.