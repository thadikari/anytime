from . import registry
import utilities.data as du
import utilities.models.cifar as cifar
import utilities.models.mnist as mnist
import utilities.models.alexnet as alex
import utilities.models.wrnet as wrnet
import utilities.data.imagenet as imagenet




@registry.register('cifar10')
def create_cifar(batch_size, test_size=-1):
    placeholders, feed_ds, init_call = du.get_dataset_pipeline('cifar10', batch_size, test_size)
    kwargs, feed_model = cifar.create_plh(with_data=False)

    class ModelFac:
        def __call__(self, feature, target):
            logits = cifar.create_conv10(feature, **kwargs)
            self.accuracy, sum_loss, self.avg_loss = du.compute_metrics(logits, target)
            return sum_loss
        def get_metrics(self): return self.accuracy, self.avg_loss

    return placeholders, ModelFac(), du.merge_feed_dicts(feed_ds, feed_model), init_call




def create_mnist(ds_name):
    def mnist_inner(batch_size, test_size=-1):

        class ModelFac:
            def __call__(self, feature, target):
                logits = mnist.create_conv(feature)
                self.accuracy, sum_loss, self.avg_loss = du.compute_metrics(logits, target)
                return sum_loss
            def get_metrics(self): return self.accuracy, self.avg_loss

        placeholders, feed_ds, init_call = du.get_dataset_pipeline(ds_name, batch_size, test_size)
        return placeholders, ModelFac(), feed_ds, init_call
    return mnist_inner

register = lambda name: registry.register(name)(create_mnist(name))
register('fashion_mnist')
register('mnist')




def create_imagenet(ds_name, model):
    def imagenet_inner(batch_size, test_size=1024):
        placeholders, feed_ds, init_call = imagenet.get_dataset_pipeline(ds_name, batch_size, test_size)
        kwargs, feed_model = model.create_plh(with_data=False)

        class ModelFac:
            def __call__(self, feature, target):
                logits = model.create_model(feature, **kwargs)
                self.accuracy, sum_loss, self.avg_loss = du.compute_metrics(logits, target)
                return sum_loss
            def get_metrics(self): return self.accuracy, self.avg_loss

        return placeholders, ModelFac(), du.merge_feed_dicts(feed_ds, feed_model), init_call
    return imagenet_inner

regimgnt = lambda name, ds_name, model=wrnet: registry.register(name)(create_imagenet(ds_name, model))
for vv in [8,16,32,64]: regimgnt(f'imagenet{vv}', f'imagenet_resized/{vv}x{vv}')
regimgnt('imagenet', 'imagenet2012')
regimgnt('alexnet', 'imagenet2012', alex)
