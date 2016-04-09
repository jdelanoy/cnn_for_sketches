#!/bin/sh

n_max=6
cnt=0
for i in {73..73}
do
    echo "****************"$i
    #echo $name
    python build_modelnet_par_database.py $i &
    [[ $((i%n_max)) -eq 0 ]] && wait
done
