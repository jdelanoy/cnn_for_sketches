#!/bin/sh

n_max=6
cnt=0
#echo /home/rendus/CSG/out/depth/1000_7.png
while [ ! -f /home/rendus/CSG/out/depth/1000_7.png ]
do
    sleep 100
    echo "wait"
done
python build_CSG_database.py

for i in {1..99}
do
    echo "****************"$i
    #echo $name
    echo /home/rendus/CSG/out/depth/$(((i+1)*1000))_7.png
    while [ ! -f /home/rendus/CSG/out/depth/$(((i+1)*1000))_7.png ]
    do
	sleep 100
	echo "wait"
    done
    python build_CSG_database.py $i
    #[[ $((i%n_max)) -eq 0 ]] && wait

done
