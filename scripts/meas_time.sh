#!/usr/bin/env bash


if [ -f ./report.csv ]
then
    rm ./report.csv
fi

for proc in $(seq $(nproc))
do
    /usr/bin/time -a -o ./report.csv -f "$proc,%e" mpiexec -n $proc ./build/mpi/cellular_automat 10000 110 1000
done