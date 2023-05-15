#!/bin/bash
max_rep=30
max_bids=50
max_edges=9
timeout=100  # Set a timeout of 300 seconds (5 minutes)


rm -rf "res"
mkdir -p "res"


# for i in $(seq 100 100 1000)
# for i in $(seq 1 1 1)
for i in $(seq 0 0.1 1); # vary alpha
do
    rm -rf "res/""$i"
    mkdir -p ./"res"/"$i"/
    for b in `seq 0 $max_rep`
    do
        # Start the Python process and get its PID
        python3 main.py "$max_edges" "$i" > "./res/$i/$b" &        pid=$!

        # Wait for the process to complete or time out
        start_time=$(date +%s)
        while ps -p $pid > /dev/null && [ $(( $(date +%s) - $start_time )) -lt $timeout ]; do sleep 1; done

        # Check if the process has completed or timed out
        if ps -p $pid > /dev/null; then
            # The process has timed out, kill it
            kill $pid
            echo "Process $pid has timed out and been killed"
        fi

        # Move debug log to result directory
        mv debug.log ./"res"/"$i"/debug_"$b".log

        # Print the iteration number
        echo "$i"
    done
done