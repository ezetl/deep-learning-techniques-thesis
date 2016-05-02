#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12


ROOT=/home/$USER
CAFFE=/home/$USER/Software/caffe/
TOOLS=${CAFFE}build/tools
DATA=./data
LMDB=/media/ezetl/0C74D0DD74D0CB1A/mnist/mnist_train_lmdb

$TOOLS/compute_image_mean $LMDB \
  $DATA/mean_mnist.binaryproto


echo "Done."
