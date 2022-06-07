#!/bin/bash

# source /ereborfs/$USER/spkl/bin/activate 

while getopts r:s: flag
do
    case "${flag}" in
        r) red=${OPTARG};;
		s) seed=${OPTARG};;
    esac
done

type="random_sim.py"

BIN="C:/Users/guill/Documents/MEGA/Documents/Master/S3/MT/ConvexSNNs/$type"
RESDIR="C:/Users/guill/Documents/MEGA/Documents/Master/S3/MT/ConvexSNNs/saved_bbox/seed$seed/"
mkdir -p $RESDIR

echo "Redundancy: $red"

NUM_DIM=4

ARGS="--seed $seed --dir $RESDIR"

for d in `seq 1 1 $NUM_DIM`; do
	dim="$(python -c "print (2** ($d+1))")"
	n="$(python -c "print ($red*$dim)")"
	echo "Neurons: $n"
	nice python $BIN $ARGS --dimension $dim --nb_neurons $n
done
