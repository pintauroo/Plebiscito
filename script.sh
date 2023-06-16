#!/bin/bash
max_rep=30
num_edges=10
req_number=100
timeout=200  # Set a timeout of 300 seconds (5 minutes)
pid=0

trap on_sigint SIGINT
on_sigint() { 
    kill "-INT" "$pid"
    exit
}

strings=("stefano")
# strings=("alpha_GPU_CPU" "stefano")

rm -rf "res" > /dev/null 2>&1
rm -f "alpha_GPU_CPU.csv" > /dev/null 2>&1
rm -f "alpha_BW_CPU.csv" > /dev/null 2>&1
rm -f "alpha_GPU_BW.csv" > /dev/null 2>&1
rm -f "stefano.csv" > /dev/null 2>&1
mkdir -p "res" > /dev/null 2>&1


# for i in $(seq 100 100 1000)
# for i in $(seq 1 1 1)
for filename in "${strings[@]}"
    do
    echo $filename
    for i in $(seq 0.5 0.25 0.5)
    # for i in $(seq 1 1 10)
    do
        rm -rf "res/""$i"
        mkdir -p ./"res"/"$i"/
        for b in `seq 0 $max_rep`
        do
            echo "main.py $req_number $i $num_edges $filename"
            # Start the Python process and get its PID
            python3 main.py "$req_number" "$i" "$num_edges" "$filename" "$b"> "./res/$i/$b" &        
            pid=$!

            # Wait for the process to complete or time out
            start_time=$(date +%s)
            while ps -p $pid > /dev/null && [ $(( $(date +%s) - $start_time )) -lt $timeout ]; do sleep 1; done

            # Check if the process has completed or timed out
            if ps -p $pid > /dev/null; then
                # The process has timed out, kill it
                kill "-INT" "$pid" 
                echo "Process $pid has timed out and been killed"
            fi

            # Move debug log to result directory
            mv debug.log ./"res"/"$i"/debug_"$b".log

            # Print the iteration number
            echo "rep: $b,  alpha: $i"
        done
    done
done