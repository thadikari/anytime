from . import registry
import utilities.data as du
import utilities.models.cifar as cifar
import utilities.models.mnist as mnist
import utilities.models.alexnet as alex
import utilities.models.wrnet as wrnet
import utilities.data.imagenet as imagenet




@registry.register('cifar10')
def create_cifar(dataset_args):
    ds = du.make_dataset('cifar10', dataset_args)
    kwargs, feed_model = cifar.create_plh(with_data=False)
    create_cifar.train_num_examples = ds.train_num_examples

    class ModelFac:
        def __call__(self, feature, target):
            logits = cifar.create_conv10(feature, **kwargs)
            self.accuracy, sum_loss, self.avg_loss = du.compute_metrics_ex(logits, target)
            return sum_loss
        def get_metrics(self): return self.accuracy, self.avg_loss

    fds = du.merge_feed_dicts((ds.get_train_fd, ds.get_test_fd), feed_model)
    return ds.placeholders, ModelFac(), fds, ds.init




def create_mnist(ds_name):
    def mnist_inner(dataset_args):

        class ModelFac:
            def __call__(self, feature, target):
                logits = mnist.create_conv(feature)
                self.accuracy, sum_loss, self.avg_loss = du.compute_metrics_ex(logits, target)
                return sum_loss
            def get_metrics(self): return self.accuracy, self.avg_loss

        ds = du.make_dataset(ds_name, dataset_args)
        mnist_inner.train_num_examples = ds.train_num_examples
        return ds.placeholders, ModelFac(), (ds.get_train_fd, ds.get_test_fd), ds.init
    return mnist_inner

register = lambda name: registry.register(name)(create_mnist(name))
register('fashion_mnist')
register('mnist')




def create_imagenet(ds_name, model):
    def imagenet_inner(dataset_args):
        ds = imagenet.make_dataset(ds_name, dataset_args)
        imagenet_inner.train_num_examples = ds.train_num_examples
        kwargs, feed_model = model.create_plh(with_data=False)

        class ModelFac:
            def __call__(self, feature, target):
                logits = model.create_model(feature, **kwargs)
                self.accuracy = du.compute_accuracy_topk(logits, target, 5)
                sum_loss, self.avg_loss = du.compute_losses_ex(logits, target)
                return sum_loss
            def get_metrics(self): return self.accuracy, self.avg_loss

        fds = du.merge_feed_dicts((ds.get_train_fd, ds.get_test_fd), feed_model)
        return ds.placeholders, ModelFac(), fds, ds.init
    return imagenet_inner

regimgnt = lambda name, ds_name, model=wrnet: registry.register(name)(create_imagenet(ds_name, model))
for vv in [8,16,32,64]: regimgnt(f'imagenet{vv}', f'imagenet_resized/{vv}x{vv}')
regimgnt('imagenet', 'imagenet2012')
regimgnt('alexnet', 'imagenet2012', alex)
