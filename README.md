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
* Create an MPI cluster - [StarCluster](http://star.mit.edu/cluster/docs/latest/installation.html) may be helpful.
* Sample commands:
``` shell
mpi1 python -u run_eval.py mnist fmb rms 64
mpi4 python -u run_eval.py mnist amb adm 64 --amb_time_limit 9.2 --amb_num_splits 64 --starter_learning_rate 0.001
mpi4 python -u run_eval.py cifar10 amb adm 64 --amb_time_limit 9.2 --amb_num_splits 64 --starter_learning_rate 0.001 --test_size 100
mpi4 python -u run_eval.py mnist amb rms 4096 --amb_time_limit 9.2 --amb_num_splits 64 --starter_learning_rate 0.001 --induce
mpiall python -u run_eval.py mnist amb rms 1024 --amb_time_limit 1.9 --amb_num_splits 16
mpi11 python -u run_eval.py cifar10 amb rms 256 --amb_time_limit 5.5 --amb_num_splits 16 --test_size 100 --induce > ~/checkpoints/output_amb 2>&1
mpi11 python -u run_eval.py cifar10 fmb rms 256 --test_size 100 --induce > ~/checkpoints/output_fmb 2>&1
```
* Here, `mpi1`, `mpi4` and `mpiall` are aliases. For example `mpi4` translates to `mpirun -host master,node001,node002,node003`. 
* For CIFAR10 it is important to set a low value for `test_size`. Otherwise master will use all 10,000 samples in the test dataset to evaluate the model. As a result workers will have to wait to send updates to the master. 
* A sample log line printed by a worker looks like `Sending [256] examples, compute_time [5.63961], last_idle [0.267534], last_send [0.244859]`.
    * `last_send`: in the last round, time spent sending the update to the master.
    * `last_idle`: in the last round, time spent after sending an update till starting computations for the next round (includes receiving time from the master as well).

## Stats on EC2
Commands and sample worker outputs:
* `mpi11 python -u run_eval.py cifar10 fmb rms 512 --test_size 100`:
``` shell
wk10|Sending [512] examples, compute_time [11.353], last_idle [0.297866], last_send [0.271033]
wk4|Sending [512] examples, compute_time [11.3975], last_idle [0.278468], last_send [0.255518]
wk0|step = 9, loss = 4.4926357, learning_rate = 0.001, accuracy = 0.13 (11.885 sec)
```
* `mpi11 python -u run_eval.py cifar10 amb rms 512 --amb_time_limit 11 --amb_num_splits 16 --test_size 100`:
``` shell
wk8|Sending [512] examples, compute_time [11.485], last_idle [0.765861], last_send [0.25578]
wk4|Sending [512] examples, compute_time [11.4716], last_idle [0.777958], last_send [0.247732]
wk0|loss = 3.5509295, learning_rate = 0.001, step = 20, accuracy = 0.09 (12.469 sec)
```
* `mpi11 python -u run_eval.py cifar10 fmb rms 256 --test_size 100`
```
wk4|Sending [256] examples, compute_time [5.64347], last_idle [0.241176], last_send [0.221801]
wk8|Sending [256] examples, compute_time [5.66594], last_idle [0.258161], last_send [0.421286]
wk0|step = 109, loss = 2.3923714, learning_rate = 0.001, accuracy = 0.13 (6.153 sec)
```
* `mpi11 python -u run_eval.py cifar10 amb rms 256 --amb_time_limit 5.0 --amb_num_splits 8 --test_size 100`
```
wk5|Sending [256] examples, compute_time [5.69975], last_idle [0.257738], last_send [0.347983]
wk3|Sending [256] examples, compute_time [5.71114], last_idle [0.250323], last_send [0.344623]
wk0|step = 129, learning_rate = 0.001, loss = 2.265991, accuracy = 0.15 (6.426 sec)
```


### Effect of splitting minibatches using `tf.while_loop`
* AMB implementation in this code uses `tf.while_loop` to split minibatches.
* The input minibatch is split into `amb_num_splits` 'micro' batches, each of size `batch_size/amb_num_splits`. The gradients of splits are then calculated in a loop, starting from the first while the elapsed time>`amb_time_limit`. When the condition fails the worker sends the gradients (summed across the processed splits) to master.
* The execution speed for `amb_num_splits=10` is lower than that for `amb_num_splits=1` even for the same `batch_size`. Can measure execution speed drop on different platforms (EC2, Compute Canada), NN architectures (fully-connected, convolutional). 
* Following plots are produced using [`src/test_slices.py`](src/test_slices.py) which includes data generating and plotting commands.
* The CIFAR10 model used in this code produces following output on EC2.
    * Number of splits: `amb_num_splits`
    * Split size: `batch_size`/`amb_num_splits`
    * Time per step: Time taken to go through all the splits (covering the whole batch)
    * Time per sample: Time per step divided by batch size
<img src="data/test_slices/cifar10_ec2-m3-xlarge.png?raw=true"/>

* Conclusion: For CIFAR10, if `batch_size` > 512, maintaining a split size > 32 (2^5) will cause a minimal impact on the execution time. 
* This means for `batch_size`=512 set `amb_num_splits`=512/32=16.
* Below is another example for fully connected (top) vs convolutional (bottom) network for a toy dataset. Note that the while loop has a lower impact for convolutional nets. This is because the matrix multiplication in fully connected nets is well supported in modern hardware.
* See more in [`data/test_slices`](data/test_slices).

<img src="data/test_slices/toy_model_fc_ec2-t2-micro.png?raw=true"/>
<img src="data/test_slices/toy_model_conv_ec2-t2-micro.png?raw=true"/>

* Sample commands:
``` shell
python -u test_slices.py main --batch_size 64 --num_splits 2 --model mnist
python -u test_slices.py main --batch_size 64 --num_splits 2 --model cifar10
```
