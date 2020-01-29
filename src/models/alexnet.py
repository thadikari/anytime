import tensorflow as tf
import numpy as np


layers = tf.layers

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
DROPOUT_KEEP_PROB = 0.5
FC_HIDDEN_SIZE = 4096
K_BIAS = 2
N_DEPTH_RADIUS = 5
ALPHA = 1e-4
BETA = 0.75

def make_model(feature, target):

    # https://github.com/gholomia/AlexNet-Tensorflow/blob/master/src/alexnet.py
    # https://www.learnopencv.com/wp-content/uploads/2018/05/AlexNet-1.png

    vv = lambda shape: tf.Variable(lambda: tf.truncated_normal(shape=shape, mean=0, stddev=0.08))

    # Convolution Layer 1 | Response Normalization | Max Pooling | ReLU
    c_layer_1 = tf.nn.conv2d(feature, vv([11,11,3,96]), strides=[1, 4, 4, 1], padding="VALID", name="c_layer_1")
    c_layer_1 = tf.nn.relu(c_layer_1)
    # c_layer_1 = tf.nn.lrn(c_layer_1, depth_radius=N_DEPTH_RADIUS, bias=K_BIAS, alpha=ALPHA, beta=BETA)
    c_layer_1 = tf.nn.max_pool(c_layer_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # Convolution Layer 2 | Response Normalization | Max Pooling | ReLU
    c_layer_2 = tf.nn.conv2d(c_layer_1, vv([5,5,96,256]), strides=[1, 1, 1, 1], padding="SAME", name="c_layer_2")
    c_layer_2 = tf.nn.relu(c_layer_2)
    # c_layer_2 = tf.nn.lrn(c_layer_2, depth_radius=5, bias=K_BIAS, alpha=ALPHA, beta=BETA)
    c_layer_2 = tf.nn.max_pool(c_layer_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # Convolution Layer 3 | ReLU
    c_layer_3 = tf.nn.conv2d(c_layer_2, vv([3,3,256,384]), strides=[1, 1, 1, 1], padding="SAME", name="c_layer_3")
    c_layer_3 = tf.nn.relu(c_layer_3)

    # Convolution Layer 4 | ReLU
    c_layer_4 = tf.nn.conv2d(c_layer_3, vv([3,3,384,384]), strides=[1, 1, 1, 1], padding="SAME", name="c_layer_4")
    c_layer_4 = tf.nn.relu(c_layer_4)

    # Convolution Layer 5 | ReLU | Max Pooling
    c_layer_5 = tf.nn.conv2d(c_layer_4, vv([3,3,384,256]), strides=[1, 1, 1, 1], padding="SAME", name="c_layer_5")
    c_layer_5 = tf.nn.relu(c_layer_5)
    c_layer_5 = tf.nn.max_pool(c_layer_5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # Flatten the multi-dimensional outputs to feed fully connected layers
    # feature_map = tf.reshape(c_layer_5, [-1, 13*13*256], name="myreshape")
    feature_map = tf.contrib.layers.flatten(c_layer_5)

    fc = tf.contrib.layers.fully_connected
    # Fully Connected Layer 1 | Dropout
    fc_layer_1 = fc(inputs=feature_map, num_outputs=4096, activation_fn=tf.nn.relu)
    fc_layer_1 = tf.nn.dropout(fc_layer_1, DROPOUT_KEEP_PROB)

    # Fully Connected Layer 2 | Dropout
    fc_layer_2 = fc(inputs=fc_layer_1, num_outputs=4096, activation_fn=tf.nn.relu)
    fc_layer_2 = tf.nn.dropout(fc_layer_2, keep_prob=DROPOUT_KEEP_PROB)

    # Fully Connected Layer 3 | Softmax
    fc_layer_3 = fc(inputs=fc_layer_2, num_outputs=1000, activation_fn=None)
    cnn_output = tf.nn.softmax(fc_layer_3)
    # print(fc_layer_3)
    # exit()

    logits = fc_layer_3
    target_1h = tf.one_hot(tf.cast(target, tf.int32), 1000)

    loss = tf.compat.v1.losses.softmax_cross_entropy(target_1h, logits)
    predict = tf.argmax(logits, 1)
    is_correct = tf.equal(tf.cast(target, 'int64'), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, 'float'))
    # error = 100*(1 - tf.reduce_mean(tf.cast(self.is_correct, 'float')))
    # print([aa.get_shape().as_list() for aa in tf.trainable_variables()])
    return accuracy, loss

def create_data(batch_size):
    return np.random.normal(size=[batch_size, 227, 227, 3]),\
           np.random.choice(1000, batch_size)

def input_generator(batch_size):
    while 1: yield create_data(batch_size)


def get_fac_elements(batch_size, test_size=-1):
    class ModelFac:
        def __call__(self, feature, target):
            self.accuracy, self.loss = make_model(feature, target)
            return self.loss

        def get_metrics(self):
            return self.accuracy, self.loss

    image = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
    label = tf.placeholder(tf.float32, shape=[None])
    placeholders = image, label

    generator = input_generator(batch_size)
    x_test, y_test = create_data(100)
    def get_train_fd(): return dict(zip([image, label], next(generator)))
    return placeholders, ModelFac(), get_train_fd, lambda: {image:x_test, label:y_test}
