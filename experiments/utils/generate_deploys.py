#!/usr/bin/env python2.7
from nets.cnn_factory import KITTINetFactory, MNISTNetFactory 

imagenet, _, _ = KITTINetFactory.standar(
        lmdb_path='./deploys',
        num_classes=1000,
        is_train=False,
        learn_all=False,
        is_imagenet=True
        )
with open('deploys/imagenet.prototxt', 'w') as f:
    f.write(str(imagenet.to_proto()))

kitti, _, _ = KITTINetFactory.standar(
        lmdb_path='./deploys',
        num_classes=1000,
        is_train=False,
        learn_all=False,
        is_imagenet=False
        )
with open('deploys/kitti.prototxt', 'w') as f:
    f.write(str(kitti.to_proto()))

mnist, _, _ = MNISTNetFactory.standar(
        lmdb_path='./deploys',
        scale=1/255.0,
        is_train=False,
        learn_all=False
        )
with open('deploys/mnist.prototxt', 'w') as f:
    f.write(str(mnist.to_proto()))
