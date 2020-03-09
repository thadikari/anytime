import tensorflow as tf

from . import registry
import utilities.data as du
import utilities.models.cifar as model


@registry.register('cifar10')
def get_fac_elements(batch_size, test_size=-1):

    x, y, keep_prob = model.create_plh()

    class ModelFac:
        def __call__(self, feature, target):
            self.accuracy, sum_loss, self.avg_loss = du.compute_metrics(model.create_conv10(feature, keep_prob), target, 10)
            return sum_loss

        def get_metrics(self):
            return self.accuracy, self.avg_loss

    # accuracy, loss = make_model(x,y,keep_prob)
    (x_train, y_train), (x_test, y_test) = du.get_dataset('cifar10')

    generator = du.input_generator(x_train, y_train, batch_size)
    if test_size>=0:
        x_test, y_test = du.permute(x_test, y_test, seed=test_size)
        x_test, y_test = x_test[:test_size], y_test[:test_size]
    def get_train_fd():
        x_, y_ = next(generator)
        return {x:x_, y:y_, keep_prob:0.7}
    return (x,y), ModelFac(), get_train_fd, lambda: {x:x_test, y:y_test, keep_prob:1.0}
