# Anytime mini-batch implementation on Tensorflow

### Generating data and plots
* [`src/run_eval.py`](src/run_eval.py): generates data (see therewithin the applicable arguments).
* [`src/graph.py`](src/graph.py): plots data (see below for a sample).

### Sample comparison of Anytime and Fixed mini-batch (AMB and FMB)
* m3.xlarge instances in Amazon EC2
* Hub-and-spoke - 10 nodes and master
* CIFAR10 dataset
* Induced stragglers
* RMS-prop optimizer
* AMB time limit = 5.5s
* FMB batchsize = 256
* See more samples in [`data`](data).

<img src="data/600_cifar10_v4/cifar10__set3/all_plots.png?raw=true"/>

### Effect of splitting minibatches using `tf.while_loop`
* [`src/test_slices.py`](src/test_slices.py) includes data generating and plotting commands.
* AMB implementation in this code uses `tf.while_loop` to split minibatches.
* Can measure how much execution speed drops on different platforms (EC2, Compute Canada), NN architectures (fully-connected, convolutional). 
* See below for a sample: fully connected (top) vs convolutional (bottom) network. 
* More in [`data/test_slices`](data/test_slices).

<img src="data/test_slices/toy_model_fc_ec2-t2-micro.png?raw=true"/>
<img src="data/test_slices/toy_model_conv_ec2-t2-micro.png?raw=true"/>
