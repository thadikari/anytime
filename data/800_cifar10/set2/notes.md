# run commit hash
1209b78290c868bcd4651a47e84adf0dd6b35c84

# commands
mpi11 python -u run_eval.py cifar10 amb rms 356 --amb_time_limit 6.2 --amb_num_splits 16 --test_size 100  --induce --decay_rate 0.93 > $SCRATCH/anytime/output_amb 2>&1
mpi11z python -u run_eval.py cifar10 fmb rms 242 --amb_time_limit 6.2 --amb_num_splits 16 --test_size 100 --induce --decay_rate 0.93 > $SCRATCH/anytime/output_fmb 2>&1