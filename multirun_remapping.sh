#!/bin/bash

# source /ereborfs/$USER/spkl/bin/activate 

while getopts e: flag
do
    case "${flag}" in
        e) encod=${OPTARG};;
    esac
done

BIN="run_pathint"
RESDIR="./data/TestVariance/"
mkdir -p $RESDIR

NUM_VARIANCE=5
NUM_ENV=4

DATARGS="--dim_pcs 2 --nb_neurons 512 --encoding $encod --dim_bbox 16 --model randclosed-load-polyae 
--save --decoder_amp 0.3 --thresh_amp 1.2"
ARGS="--dir $RESDIR"


for sigid in `seq 0 1 $NUM_VARIANCE`; do
	sigma="$(python -c "print (1/$NUM_VARIANCE * $sigid)")"
	echo "Variance: $sigma"
	for env in `seq 0 1 $(($NUM_ENV-1))`; do
		echo "Env: $env"
		nice python -m $BIN $DATARGS $ARGS --env $env --embedding_sigma $sigma
	done
done

