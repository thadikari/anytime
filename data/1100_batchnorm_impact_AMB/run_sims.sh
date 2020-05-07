
if [ 1 ] # [  ] to run else section
then # imagenet32
    EXEC="python -u run_perf_amb.py imagenet32"
    ARGS="128 --amb_time_limit 15.0 --test_size 128 --master_cpu --last_step 2000 --starter_learning_rate 0.1"
else # cifar10
    EXEC="python -u run_perf_amb.py cifar10"
    ARGS="256 --amb_time_limit 10.2 --test_size 128 --master_cpu --last_step 1500"
fi

log () { echo "$1"; }
run () { log ">> $1"; eval "$1"; }
exc () { run "mpirun -n 2 $EXEC amb mom $ARGS --amb_num_partitions $1"; }

for i in 2 4 8 16 32 64 128; do exc $i; done
run "$EXEC fmb mom $ARGS"
run "mpirun -n 2 $EXEC fmb mom $ARGS"
