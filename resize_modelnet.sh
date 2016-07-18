#!/bin/sh

n_max=24

#mkdir /home/rendus/modelnet/out/test/{normal,cont,depth,T5} -p

path=/home/rendus/modelnet/test/depth
cnt=0
for img in $(ls $path/)
do
    python resize_modelnet.py $img &
    let cnt+=1
    [[ $((cnt%n_max)) -eq 0 ]] && echo $cnt
    [[ $((cnt%n_max)) -eq 0 ]] && wait
done

for 
