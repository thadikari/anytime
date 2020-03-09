import tensorflow as tf

from . import registry
import utilities.data as du
import utilities.models.mnist as model


def get_fac_elements_outer(dataset_name):
    def get_fac_elements_inner(batch_size, test_size=-1):

        class ModelFac:
            def __call__(self, feature, target):
                self.accuracy, sum_loss, self.avg_loss = du.compute_metrics(model.create_conv(feature), target, 10)
                return sum_loss

            def get_metrics(self):
                return self.accuracy, self.avg_loss

        placeholders = model.create_plh()
        image, label = placeholders
        (x_train, y_train), (x_test, y_test) = du.get_dataset(dataset_name)

        generator = du.input_generator(x_train, y_train, batch_size)
        if test_size>=0:
            x_test, y_test = du.permute(x_test, y_test, seed=test_size)
            x_test, y_test = x_test[:test_size], y_test[:test_size]
        def get_train_fd():
            return dict(zip([image, label], next(generator)))
        return placeholders, ModelFac(), get_train_fd, lambda: {image:x_test, label:y_test}
    return get_fac_elements_inner


register = lambda name: registry.register(name)(get_fac_elements_outer(name))
register('fashion_mnist')
register('mnist')
