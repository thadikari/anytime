# Tensorflow-based Anytime mini-batch implementation

### Comparison of Anytime and Fixed mini-batch (AMB and FMB)
* Run on m3.xlarge instances in Amazon EC2
* Hub-and-spoke - 10 nodes and master
* Induced stragglers
* RMS-prop optimizer
* FMB batchsize = 128

<img src="data/600_cifar10_v4/cifar10__set3/all_plots.png?raw=true"/>

### Generating data and plots
* Execute [`src/run_eval.py`](src/run_eval.py) to generate data (see therewithin the applicable arguments).
* Execute [`src/graph.py`](src/graph.py) to plot data.

### Split minibatches using `tf.while_loop`
* Log speed of TF code when minibatches are split using `tf.while_loop`.
* Compare the execution speed on different platforms (EC2, Compute Canada), NN architectures (fully-connected, convolutional). 
* [`src/test_slices.py`](src/test_slices.py) includes data generating and plotting commands. 
* See samples in [`data/test_slices`](data/test_slices).

<img src="data/test_slices/toy_model_conv_ec2-t2-micro.png?raw=true" />
<img src="data/test_slices/toy_model_fc_ec2-t2-micro.png?raw=true" />
