#!/bin/bash
max_rep=30
# num_edges=4
# req_number=1
timeout=60  # Set a timeout of 300 seconds (5 minutes)
pid=0

trap on_sigint SIGINT
on_sigint() { 
    kill "-INT" "$pid"
    exit
}

strings=("alpha_GPU_CPU" "zio_alessandro" "alpha_BW_CPU" "alpha_GPU_BW" "stefano")
# strings=("zio_alessandro" "alpha_BW_CPU" "alpha_GPU_BW" "stefano")
# strings=("stefano")

rm -rf "res" > /dev/null 2>&1
rm -f "alpha_GPU_CPU.csv" > /dev/null 2>&1
rm -f "alpha_BW_CPU.csv" > /dev/null 2>&1
rm -f "alpha_GPU_BW.csv" > /dev/null 2>&1
rm -f "stefano.csv" > /dev/null 2>&1
rm -f "zio_alessandro.csv" > /dev/null 2>&1
mkdir -p "res" > /dev/null 2>&1
mkdir -p "convergence"

filename="alpha_GPU_CPU"
i=0

# for filename in "${strings[@]}"; do
for req_number in $(seq 1 20); do
    for layer_num in $(seq 1 20); do

        echo $filename
        # for i in $(seq 0.0 0.5 1); do
        for num_edges in $(seq 2 20); do
            
            mkdir -p ./"res"/"$filename"/"$i"/
            for b in `seq 0 $max_rep`; do
                while true; do

                    echo "main.py $req_number $i $num_edges $filename"
                    # Start the Python process and get its PID
                    python3 main.py "$req_number" "$i" "$num_edges" "$filename" "$layer_num"> "./res/"$filename"/"$i"/$num_edges"_"$layer_num"_"$b" &        
                    pid=$!

                    # Wait for the process to complete or time out
                    start_time=$(date +%s)
                    while ps -p $pid > /dev/null && [ $(( $(date +%s) - $start_time )) -lt $timeout ]; do sleep 1; done

                    # Check if the process has completed or timed out
                    if ps -p $pid > /dev/null; then
                        # The process has timed out, kill it
                        kill "-INT" "$pid" 
                        mv debug.log ./"res"/"$filename"/"$i"/"$b"_FAIL.log
                        mv ./res/"$filename"/"$i"/"$b" ./res/"$filename"/"$i"/"$b"_FAIL
                        echo "Process $pid has timed out and been killed, retrying..."
                    else
                        # Move debug log to result directory
                        mv debug.log ./"res"/"$filename"/"$i"/"$num_edges"_"$layer_num"_"$b".log

                        # Print the iteration number
                        echo "rep: $b, alpha: $i"
                        break  # Exit the retry loop if the process completed successfully
                    fi
                done
            done
            mv 'alpha_GPU_CPU.csv' "convergence"/"$num_edges"_"$req_number"_"$layer_num"
    
        done

    done
done