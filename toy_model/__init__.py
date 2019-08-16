import tensorflow as tf
import numpy as np


layers = tf.layers

def make_model_fc(feature, target):
    target_1h = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)
    l1 = layers.dense(feature, 500, activation=tf.nn.relu)
    l2 = layers.dense(l1, 1000, activation=tf.nn.relu)
    l3 = layers.dense(l2, 500, activation=tf.nn.relu)
    logits = layers.dense(l3, 10, activation=None)
    # loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(target_1h, logits, reduction='none'))
    loss = tf.compat.v1.losses.softmax_cross_entropy(target_1h, logits)
    predict = tf.argmax(logits, 1)
    is_correct = tf.equal(tf.cast(target, 'int64'), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, 'float'))
    # error = 100*(1 - tf.reduce_mean(tf.cast(self.is_correct, 'float')))
    return accuracy, loss

make_model_fc.shapes = [(100,500), (500), (500,1000), (1000), (1000,500), (500), (500,10), (10)]

def make_model_conv(feature, target):
    target_1h = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)
    feature = tf.reshape(feature, [-1, 10, 10, 1])

    with tf.compat.v1.variable_scope('conv_layer1'):
        h_conv1 = layers.conv2d(feature, 32, kernel_size=[5, 5],
                                activation=tf.nn.relu, padding="SAME")
        h_pool1 = tf.nn.max_pool(
            h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second conv layer will compute 64 features for each 5x5 patch.
    with tf.compat.v1.variable_scope('conv_layer2'):
        h_conv2 = layers.conv2d(h_pool1, 64, kernel_size=[5, 5],
                                activation=tf.nn.relu, padding="SAME")
        h_pool2 = tf.nn.max_pool(
            h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 3 * 3 * 64])

    # Densely connected layer with 1024 neurons.
    logits = layers.dense(h_pool2_flat, 10, activation=None)
    # loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(target_1h, logits, reduction='none'))
    loss = tf.compat.v1.losses.softmax_cross_entropy(target_1h, logits)
    predict = tf.argmax(logits, 1)
    is_correct = tf.equal(tf.cast(target, 'int64'), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, 'float'))
    # error = 100*(1 - tf.reduce_mean(tf.cast(self.is_correct, 'float')))
    # print(tf.trainable_variables())
    return accuracy, loss

make_model_conv.shapes = [(5,5,1,32), (32), (5,5,32,64), (64), (576,10), (10)]

def create_data(batch_size):
    return np.random.normal(size=[batch_size,100]),\
           np.random.choice(10, batch_size)

def input_generator(batch_size):
    while 1: yield create_data(batch_size)

def get_fac_elements(batch_size, test_size=-1):

    make_model = make_model_conv    # make_model_conv | make_model_fc

    class ModelFac:
        def __call__(self, feature, target):
            self.accuracy, self.loss = make_model(feature, target)
            return self.loss
            
        def get_var_shapes(self):
            return make_model.shapes

        def get_metrics(self):
            return self.accuracy, self.loss

    placeholders = tf.placeholder(tf.float32, [None, 100], name='image'),\
                   tf.placeholder(tf.float32, [None], name='label')
    image, label = placeholders
    generator = input_generator(batch_size)
    x_test, y_test = create_data(100)
    def get_train_fd():
        return dict(zip([image, label], next(generator)))
    return placeholders, ModelFac(), get_train_fd, lambda: {image:x_test, label:y_test}
