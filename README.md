# Anytime mini-batch implementation on Tensorflow

### Generating data and plots
* [`src/run_eval.py`](src/run_eval.py): generates data (see therewithin the applicable arguments).
* [`src/graph.py`](src/graph.py): plots data (see below for a sample).

## Sample comparison of Anytime and Fixed mini-batch (AMB and FMB)
* m3.xlarge instances in Amazon EC2
* Hub-and-spoke - 10 nodes and master
* CIFAR10 dataset
* Induced stragglers
* RMS-prop optimizer
* AMB time limit = 5.5s
* FMB batchsize = 256
* See more samples in [`data`](data).

<img src="data/600_cifar10_v4/cifar10__set3/all_plots.png?raw=true"/>

## Instructions for running on Amazon EC2
* Create an mpi cluster. A easy way is to use [Starcluster](http://star.mit.edu/cluster/docs/latest/installation.html).
* Sample commands:
	```
	mpi1 python -u run_eval.py mnist fmb rms 64
	mpi4 python -u run_eval.py mnist amb adm 64 --amb_time_limit 9.2 --amb_num_splits 64 --starter_learning_rate 0.001
	mpi4 python -u run_eval.py cifar10 amb adm 64 --amb_time_limit 9.2 --amb_num_splits 64 --starter_learning_rate 0.001 --test_size 1000
	mpi4 python -u run_eval.py mnist amb rms 4096 --amb_time_limit 9.2 --amb_num_splits 64 --starter_learning_rate 0.001 --induce
	mpi4 python -u run_eval.py mnist amb rms 1024 --amb_time_limit 1.9 --amb_num_splits 16
	mpiall python -u run_eval.py cifar10 amb 5.5 16 rms --extra=set4
	mpiall python -u run_eval.py cifar10 fmb 16 16 adm --induce > ~/checkpoints/output_fmb 2>&1
	```
* Here, `mpi1`, `mpi4`, `mpiall` are an aliases. For example `mpi4` translates to `mpirun -host master,node001,node002,node003`. 


### Effect of splitting minibatches using `tf.while_loop`
* [`src/test_slices.py`](src/test_slices.py) includes data generating and plotting commands.
* AMB implementation in this code uses `tf.while_loop` to split minibatches.
* Can measure how much execution speed drops on different platforms (EC2, Compute Canada), NN architectures (fully-connected, convolutional). 
* See below for a sample: fully connected (top) vs convolutional (bottom) network. 
* More in [`data/test_slices`](data/test_slices).

<img src="data/test_slices/toy_model_fc_ec2-t2-micro.png?raw=true"/>
<img src="data/test_slices/toy_model_conv_ec2-t2-micro.png?raw=true"/>

* Sample commands:
	```
	python -u test_slices.py main --batch_size 64 --num_splits 2 --model mnist
	python -u test_slices.py main --batch_size 64 --num_splits 2 --model cifar10
	```
