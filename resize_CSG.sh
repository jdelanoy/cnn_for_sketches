#!/bin/sh

n_max=2
for i in {0..59}
do
    echo "========================================================="$i
    for v in {0..7}
    do
	file=/home/rendus/CSG/raw_test/normal/"$i"_"$v".png
	while [ ! -f $file ]
	do
	    echo "wait"
	    sleep 10
	done
	python resize_CSG.py $file &
    done
    [[ $(((i+1)%n_max)) -eq 0 ]] && wait
done
