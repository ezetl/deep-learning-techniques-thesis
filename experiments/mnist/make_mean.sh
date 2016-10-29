#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12


CAFFE=/home/$USER/.Software/caffe/build/tools
DATA=./data
LMDB=/media/eze/0F4A13791A35DD40/MNIST/mnist_finetuning_standar10000_lmdb

$CAFFE/compute_image_mean $LMDB \
  $DATA/mean_mnist.binaryproto


echo "Done."
