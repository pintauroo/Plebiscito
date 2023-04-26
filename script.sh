#!/bin/bash
max_rep=10
max_bids=50
max_edges=30
rm -rf "res"
mkdir -p "res"

for i in $(seq 1 20)
do
    rm -rf "res/""$i"
    mkdir -p ./"res"/"$i"/
    for b in `seq 0 $max_rep`
    do
        
        # start=`date +%s`
        # python3 main.py "$i" > ./"res"/"$i"/output"$b"
        python3 main.py "$i" > ./"res"/"$i"/"$b"
        mv debug.log ./"res"/"$i"/debug_"$b".log
        # end=`date +%s`

        # runtime=$((end-start))
        echo "$i"; 
        # echo $runtime
        # uniq -c ./"res"/output"$i" | [ $(wc -l) -eq 1 ] && echo 'consensus' || echo 'fail'
    done
done