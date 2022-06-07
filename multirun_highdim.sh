#!/bin/bash

# source /ereborfs/$USER/spkl/bin/activate 

while getopts p:r: flag
do
    case "${flag}" in
        p) pcs=${OPTARG};;
        r) red=${OPTARG};;
    esac
done


BIN="C:/Users/guill/Documents/MEGA/Documents/Master/S3/MT/ConvexSNNs/run_highdim.py"
RESDIR="C:/Users/guill/Documents/MEGA/Documents/Master/S3/MT/ConvexSNNs/data/RandTorusPCS$pcs/"
mkdir -p $RESDIR

echo "Place cells: $pcs"
echo "Redundancy: $red"

NUM_DIM=4
NUM_LOADID=3
NUM_DIR=6

DATARGS="--dim_pcs $pcs --model randclosed-load-polyae --input_amp 1 --noise_amp 1 --decoder_amp 0.1 --thresh_amp 1"
ARGS="--seed 666 --dir $RESDIR"

for d in `seq 1 1 $NUM_DIM`; do
	dim="$(python -c "print (2** ($d+1))")"
	n="$(python -c "print ($red*$dim)")"
	echo "Neurons: $n"
	for li in `seq 0 1 $(($NUM_LOADID-1))`; do
		echo "LoadID: $li"
		for dir in `seq 0 1 $(($NUM_DIR-1))`; do
			echo "Dir: $dir"
			nice python $BIN $DATARGS $ARGS --nb_neurons $n --dim_bbox $dim --input_dir $dir --load_id $li
		done
	done
done
