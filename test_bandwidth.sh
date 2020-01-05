#!/bin/bash -i

EXEC="mpimstw"
ARGS1="python -u src/run_perf_amb.py cifar10 fmb sgd 128 --last_step 100 --test_size 1"
ARGS2="python -u src/test_bandwidth.py"

log () { echo "$1"; }
run () { log ">> $1"; eval "$1"; }
exc () { run "$EXEC $1 $ARGS2 $2"; }

for len in 2**10 2**14 2**18 2**20 2**22; do
    for i in 1 2 4 8 16 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 99; do
        #echo $((len))
        exc $i "$((len))";
        sleep 5s;
    done
done
