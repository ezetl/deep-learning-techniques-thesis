#!/usr/bin/env bash

if [[ -z "$1" ]] || [[ -z "$2" ]]
then
    echo "You have to provide the source LMDB folder"
    echo "and the path where the mean file will be created."
    echo "Example:"
    echo "$0 ../datasets/Imagenet_train_lmdb ./data/mean_files/imagenet_mean.binaryproto"
    exit 0
fi

TOOLS=/home/$USER/.Software/caffe/build/tools
LMDB=$1
MEAN=$2
$TOOLS/compute_image_mean $LMDB $MEAN

echo "Done."
