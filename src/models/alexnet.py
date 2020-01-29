import tensorflow as tf
import numpy as np


layers = tf.layers

# Global dataset dictionary
dataset_dict = {
    "image_size": 224,
    "num_channels": 3,
    "num_labels": 1000,
}

# Filter shapes for each layer
conv_filter_shapes = {
    "c1_filter": [11, 11, 3, 96],
    "c2_filter": [5, 5, 48, 256],
    "c3_filter": [3, 3, 256, 384],
    "c4_filter": [3, 3, 192, 384],
    "c5_filter": [3, 3, 192, 256]
}

# Fully connected shapes
fc_connection_shapes = {
    "f1_shape": [23*23*256, 4096],
    "f2_shape": [4096, 4096],
    "f3_shape": [4096, dataset_dict["num_labels"]]
}

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

    # Weights for each layer
    conv_weights = {
        "c1_weights": tf.Variable(lambda:tf.truncated_normal(conv_filter_shapes["c1_filter"]), name="c1_weights"),
        "c2_weights": tf.Variable(lambda:tf.truncated_normal(conv_filter_shapes["c2_filter"]), name="c2_weights"),
        "c3_weights": tf.Variable(lambda:tf.truncated_normal(conv_filter_shapes["c3_filter"]), name="c3_weights"),
        "c4_weights": tf.Variable(lambda:tf.truncated_normal(conv_filter_shapes["c4_filter"]), name="c4_weights"),
        "c5_weights": tf.Variable(lambda:tf.truncated_normal(conv_filter_shapes["c5_filter"]), name="c5_weights"),
        "f1_weights": tf.Variable(lambda:tf.truncated_normal(fc_connection_shapes["f1_shape"]), name="f1_weights"),
        "f2_weights": tf.Variable(lambda:tf.truncated_normal(fc_connection_shapes["f2_shape"]), name="f2_weights"),
        "f3_weights": tf.Variable(lambda:tf.truncated_normal(fc_connection_shapes["f3_shape"]), name="f3_weights")
    }

    # Biases for each layer
    conv_biases = {
        "c1_biases": tf.Variable(lambda:tf.truncated_normal([conv_filter_shapes["c1_filter"][3]]), name="c1_biases"),
        "c2_biases": tf.Variable(lambda:tf.truncated_normal([conv_filter_shapes["c2_filter"][3]]), name="c2_biases"),
        "c3_biases": tf.Variable(lambda:tf.truncated_normal([conv_filter_shapes["c3_filter"][3]]), name="c3_biases"),
        "c4_biases": tf.Variable(lambda:tf.truncated_normal([conv_filter_shapes["c4_filter"][3]]), name="c4_biases"),
        "c5_biases": tf.Variable(lambda:tf.truncated_normal([conv_filter_shapes["c5_filter"][3]]), name="c5_biases"),
        "f1_biases": tf.Variable(lambda:tf.truncated_normal([fc_connection_shapes["f1_shape"][1]]), name="f1_biases"),
        "f2_biases": tf.Variable(lambda:tf.truncated_normal([fc_connection_shapes["f2_shape"][1]]), name="f2_biases"),
        "f3_biases": tf.Variable(lambda:tf.truncated_normal([fc_connection_shapes["f3_shape"][1]]), name="f3_biases")
    }

    dataset_dict["total_image_size"] = dataset_dict["image_size"] * dataset_dict["image_size"]

    # Declare the input and output placeholders
    img_4d_shaped = tf.reshape(feature, [-1, dataset_dict["image_size"], dataset_dict["image_size"], dataset_dict["num_channels"]])

    # Convolution Layer 1 | Response Normalization | Max Pooling | ReLU
    c_layer_1 = tf.nn.conv2d(img_4d_shaped, conv_weights["c1_weights"], strides=[1, 4, 4, 1], padding="SAME", name="c_layer_1")
    c_layer_1 += conv_biases["c1_biases"]
    c_layer_1 = tf.nn.relu(c_layer_1)
    c_layer_1 = tf.nn.lrn(c_layer_1, depth_radius=N_DEPTH_RADIUS, bias=K_BIAS, alpha=ALPHA, beta=BETA)
    c_layer_1 = tf.nn.max_pool(c_layer_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # Convolution Layer 2 | Response Normalization | Max Pooling | ReLU
    c_layer_2 = tf.nn.conv2d(c_layer_1, conv_weights["c2_weights"], strides=[1, 1, 1, 1], padding="SAME", name="c_layer_2")
    c_layer_2 += conv_biases["c2_biases"]
    c_layer_2 = tf.nn.relu(c_layer_2)
    c_layer_2 = tf.nn.lrn(c_layer_2, depth_radius=5, bias=K_BIAS, alpha=ALPHA, beta=BETA)
    c_layer_2 = tf.nn.max_pool(c_layer_2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="VALID")

    # Convolution Layer 3 | ReLU
    c_layer_3 = tf.nn.conv2d(c_layer_2, conv_weights["c3_weights"], strides=[1, 1, 1, 1], padding="SAME", name="c_layer_3")
    c_layer_3 += conv_biases["c3_biases"]
    c_layer_3 = tf.nn.relu(c_layer_3)

    # Convolution Layer 4 | ReLU
    c_layer_4 = tf.nn.conv2d(c_layer_3, conv_weights["c4_weights"], strides=[1, 1, 1, 1], padding="SAME", name="c_layer_4")
    c_layer_4 += conv_biases["c4_biases"]
    c_layer_4 = tf.nn.relu(c_layer_4)

    # Convolution Layer 5 | ReLU | Max Pooling
    c_layer_5 = tf.nn.conv2d(c_layer_4, conv_weights["c5_weights"], strides=[1, 1, 1, 1], padding="SAME", name="c_layer_5")
    c_layer_5 += conv_biases["c5_biases"]
    c_layer_5 = tf.nn.relu(c_layer_5)
    c_layer_5 = tf.nn.max_pool(c_layer_5, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="VALID")

    # Flatten the multi-dimensional outputs to feed fully connected layers
    # feature_map = tf.reshape(c_layer_5, [-1, 13*13*256], name="myreshape")
    feature_map = tf.contrib.layers.flatten(c_layer_5)
    # print(feature_map)
    # exit()

    # Fully Connected Layer 1 | Dropout
    fc_layer_1 = tf.matmul(feature_map, conv_weights["f1_weights"]) + conv_biases["f1_biases"]
    fc_layer_1 = tf.nn.dropout(fc_layer_1, keep_prob=DROPOUT_KEEP_PROB)

    # Fully Connected Layer 2 | Dropout
    fc_layer_2 = tf.matmul(fc_layer_1, conv_weights["f2_weights"]) + conv_biases["f2_biases"]
    fc_layer_2 = tf.nn.dropout(fc_layer_2, keep_prob=DROPOUT_KEEP_PROB)

    # Fully Connected Layer 3 | Softmax
    fc_layer_3 = tf.matmul(fc_layer_2, conv_weights["f3_weights"]) + conv_biases["f3_biases"]
    cnn_output = tf.nn.softmax(fc_layer_3)

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
    return np.random.normal(size=[batch_size, 224, 224, 3]),\
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

    image = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    label = tf.placeholder(tf.float32, shape=[None])
    placeholders = image, label

    generator = input_generator(batch_size)
    x_test, y_test = create_data(100)
    def get_train_fd(): return dict(zip([image, label], next(generator)))
    return placeholders, ModelFac(), get_train_fd, lambda: {image:x_test, label:y_test}
