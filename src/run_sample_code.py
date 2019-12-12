import tensorflow as tf
import numpy as np
import distributed as hvd



# run this sample using `mpirun -n 3 python -u run_sample_code.py`.


def create_model_get_sum_loss(feature, target):
    '''
    This function creates the model and returns sum of the losses of all data points.
    It is important to return sum of the losses and not the average.
    This function is not intended to be called by the user. It will be callled by AMB optimizer.
    Input should be the same number of arguments and in the same order as the `placeholders` varible 
    passed on to the `AnytimeMiniBatchDistributor.minimize` method below.
    '''
    layers = tf.layers
    target_1h = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)
    lay = layers.dense(feature, 500, activation=tf.nn.relu)
    logits = layers.dense(lay, 10, activation=None)
    losses = tf.losses.softmax_cross_entropy(onehot_labels=target_1h, logits=logits, reduction='none')
    create_model_get_sum_loss.sum_loss = tf.reduce_sum(losses)
    create_model_get_sum_loss.avg_loss = tf.reduce_mean(losses)
    return create_model_get_sum_loss.sum_loss


def create_data_for_one_iteration(batch_size):
    # Data generator for a dummy two class problem.
    class_size = int(batch_size/2)
    return np.vstack((np.random.normal(loc=1, size=[class_size,10]),
                      np.random.normal(loc=9, size=[class_size,10]))),\
           np.hstack((np.zeros(class_size), np.ones(class_size)))



batch_size = 10000
'''
This should be greater than what AMB can do within the given time limit.
Otherwise AMB will print the following warning to `tf.logging` interface in runtime.
`Increase batch_size or decrease the amb_time_limit!`
'''

amb_time_limit = 0.10
# Time limit in seconds

amb_num_splits = 20
# If `batch_size=99` and `amb_num_splits=9`, AMB will compute 99/9=11 samples at a time till timeout occurs.

learning_rate = 0.0001
is_distributed = True
tf.logging.set_verbosity(tf.logging.INFO)


x = tf.placeholder(tf.float32, shape=(None, 10), name='feature')
y =  tf.placeholder(tf.float32, shape=None, name='target')
# x,y are placeholders for holding features and labels.

global_step = tf.train.get_or_create_global_step()
opt = tf.train.GradientDescentOptimizer(learning_rate)

def create_logging_hook():
    # Print to console periodically.
    log_tn = {'step': global_step, 'loss': create_model_get_sum_loss.avg_loss}
    return tf.train.LoggingTensorHook(tensors=log_tn, every_n_iter=10)

if is_distributed:
    hvd.init()
    # Must call to initialize the mpi4py library.

    dist = hvd.AnytimeMiniBatchDistributor(opt, amb_time_limit, amb_num_splits)
    # Only the master will apply gradients using `GradientDescentOptimizer` logic.

    train_op = dist.minimize(placeholders=[x,y], cr_sum_loss=create_model_get_sum_loss, global_step=global_step)
    # `placeholders` must contain at least one element.

    hooks = [hvd.BroadcastVariablesHook(dist)]
    if hvd.is_master(): hooks.append(create_logging_hook())
    # Print logs only at the master

else:
    train_op = opt.minimize(create_model_get_sum_loss(x,y), global_step=global_step)
    hooks = [create_logging_hook()]


with tf.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
    while not mon_sess.should_stop():
        x_, y_ = create_data_for_one_iteration(batch_size)
        mon_sess.run(train_op, feed_dict={x:x_, y:y_})
