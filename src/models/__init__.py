from . import mnist, cifar10, toy_model, alexnet


reg = {'mnist': mnist.module_mnist,
       'fashion_mnist': mnist.module_fashion_mnist,
       'cifar10': cifar10,
       'alexnet': alexnet,
       'toy_model': toy_model}
