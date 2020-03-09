import tensorflow as tf
import numpy as np

from . import registry
import utilities.data as du
import utilities.models.alexnet as model


def create_data(batch_size):
    return np.random.normal(size=[batch_size, 227, 227, 3]),\
           np.random.choice(1000, batch_size)

def input_generator(batch_size):
    while 1: yield create_data(batch_size)


@registry.register('alexnet')
def get_fac_elements(batch_size, test_size=-1):
    class ModelFac:
        def __call__(self, feature, target):
            self.accuracy, sum_loss, self.avg_loss = du.compute_metrics(model.alexnet(feature, keep_prob), target, 1000)
            return sum_loss

        def get_metrics(self):
            return self.accuracy, self.avg_loss

    image, label, keep_prob = model.create_plh()
    generator = input_generator(batch_size)
    x_test, y_test = create_data(100)
    def get_train_fd():
        x_, y_ = next(generator)
        return {image:x_, label:y_, keep_prob:0.5}
    return (image, label), ModelFac(), get_train_fd, lambda: {image:x_test, label:y_test, keep_prob:1.0}
