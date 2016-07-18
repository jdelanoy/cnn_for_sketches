#!/bin/sh

n_max=2
cnt=0
#echo /home/rendus/CSG/out/depth/1000_7.png
while [ ! -f /home/rendus/CSG/out/depth/1000_7.png ]
do
    echo "wait"
    sleep 100
done
#python build_CSG_database.py


for i in {34..43}
do
    echo "****************"$i
    #echo $name
    echo /home/rendus/CSG/out/depth/$(((i+1)*1000))_7.png
    while [ ! -f /home/rendus/CSG/out/depth/$(((i+1)*1000))_7.png ]
    do
	echo "wait"
	sleep 300
    done
    python build_CSG_database.py $i 0 3 &
    [[ $(((i+1)%n_max)) -eq 0 ]] && wait

done
