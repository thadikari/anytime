same as 600_cifar10_v4 but clearer and bug fixed, specially the bug related to avg_loss and sum_loss. 

# run commit hash
adc2b6325e7f8b81430cf161d672e7456d1f3d51


# commands
mpi11 python -u run_eval.py cifar10 amb rms 356 --amb_time_limit 6.2 --amb_num_splits 16 --test_size 100  --induce
mpi11z python -u run_eval.py cifar10 fmb rms 242 --amb_time_limit 6.2 --amb_num_splits 16 --test_size 100 --induce