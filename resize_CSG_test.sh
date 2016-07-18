#!/bin/sh

n_max=8
for i in {0..59}
do
    echo "========================================================="$i
    for v in {0..7}
    do
	file=/home/rendus/CSG/raw_test/normal/test_"$i"_"$v".png
	while [ ! -f $file ]
	do
	    echo "wait"
	    sleep 10
	done
	python resize_CSG.py $file &
	[[ $(((v+1)%n_max)) -eq 0 ]] && wait
    done
done
