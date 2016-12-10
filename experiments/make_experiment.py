#!/usr/bin/env python2.7
from optparse import OptionParser, OptionGroup
import caffe
from caffe import (layers as L, params as P)
import numpy as np

caffe.set_device(0)
caffe.set_mode_gpu()


# lr and decay multipliers for conv and fc layers
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
# If you want to train the WHOLE net, then use learned_param, 
# if you want to finetune some classifier on top of the learned 
# weights, then use frozen_param
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2


def conv_relu(bottom, ks, num_out, stride=1, pad=0, group=1,
        param=learned_param, weight_filler=dict(type='gaussian', std=0.01),
        bias_filler=dict(type='constant', value=0.1)):
    """
    Creates conv+relu layers

    :param bottom: Caffe layer. The layer that feeds this convolutional layer 
    :param ks: int size of one of the kernel sides (it is square) 
    :param num_out: int total number of kernels to learn  
    :param stride: int
    :param pad: int
    :param group: int
    :param param: [dict]. Multiplying factors (lr and decay) of the weights and bias.
    :param weight_filler: dict. The weight filler can be wether 'xavier' or 'gaussian' for these experiments
    :param bias_filler: dict. Usually {'constant':1}
    :returns: Caffe conv layer, Caffe ReLU layer
    """
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
            num_output=num_out, pad=pad, group=group, param=param,
            weight_filler=weight_filler, bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)


def fc_relu(bottom, num_out, param=learned_param,
    weight_filler=dict(type='gaussian', std=0.005),
    bias_filler=dict(type='constant', value=0.1)):
    """
    Creates a fully connected+relu layer 

    :param bottom: Caffe layer. The layer that feed this fc.
    :param num_out: int total number of outputs
    :param param: [dict]. Multiplying factors (lr and decay) of the weights and bias.
    :param weight_filler: dict. The weight filler can be wether 'xavier' or 'gaussian' for these experiments
    :param bias_filler: dict. Usually {'constant':1}
    :returns: Caffe fc layer, Caffe ReLU layer
    """
    fc = L.InnerProduct(bottom, num_output=num_out, param=param,
            weight_filler=weight_filler, bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)


def max_pool(bottom, ks, stride=1):
    """
    Creates a MAX Pooling layer 
    :param bottom: Caffe layer. The layer that feed this max pooling layer.
    :param ks: int size of one of the kernel sides (it is a square) 
    :param stride: int
    :returns: Caffe Pooling layer
    """
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def alexnet(train_lmdb=None, train_labels_lmdb=None, test_lmdb=None, test_labels_lmdb=None,
        batch_size=125, scale=1.0, train=True, num_classes=1000, classifier_name='fc', 
        learn_all=False):
    """
    Creates a protoxt for the AlexNet architecture

    :param train_lmdb: str. Path to train LMDB
    :param train_labels_lmdb: str. Path to train LMDB labels
    :param test_lmdb: str. Path to train LMDB
    :param test_labels_lmdb: str. Path to test LMDB labels
    :param batch_size: int. Batch size
    :param scale: float. How to scale the images
    :param train: bool. Flag indicating if this is for deploy or training
    :param num_classes: int. number of classes for the top classifier
    :param classifier_name: str. name of the top classifier
    :param learn_all: bool. Flag indicating if we should learn all the layers from scratch
    :returns: Caffe NetSpec 
    """
    n = caffe.NetSpec()

    if train_lmdb and train_labels_lmdb:
        n.data = L.Data(name="data", top="data", include=dict(phase=caffe.TRAIN), batch_size=batch_size, backend=P.Data.LMDB, source=train_lmdb, transform_param=dict(scale=scale), ntop=1)
        n.label = L.Data(name="label", top="label", include=dict(phase=caffe.TRAIN), batch_size=batch_size, backend=P.Data.LMDB, source=train_labels_lmdb, ntop=1)
    elif train_lmdb and not train_labels_lmdb:
        n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=train_lmdb, transform_param=dict(scale=scale), ntop=2)

    if test_lmdb and test_labels_lmdb:
        n.test_data = L.Data(name="data", top="data", include=dict(phase=caffe.TEST), batch_size=batch_size, backend=P.Data.LMDB, source=test_lmdb, transform_param=dict(scale=scale), ntop=1)
        n.test_label = L.Data(name="label", top="label", include=dict(phase=caffe.TEST), batch_size=batch_size, backend=P.Data.LMDB, source=test_labels_lmdb, ntop=1)
    elif test_lmdb and not test_labels_lmdb:
        n.test_data, n.test_label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=test_lmdb, transform_param=dict(scale=scale), ntop=2)

    if not train_lmdb:   
        n.data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))

    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    elif n.label is not None:    
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    return n


if __name__ == "__main__":
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-b", "--batch-size", dest="batch_size", type="int",
            default=125, help="Batch size", metavar="INT")
    parser.add_option("-r", "--do-train", dest="train", action="store_true",
            help="If set, creates a CNN for training (i.e.: enables dropout and softmax+accuracy layers)")
    parser.add_option("-n", "--num-classes", dest="num_classes", type="int",
            default=1000, help="Number of classes of the top classifier", metavar="INT")
    parser.add_option("-a", dest="no_train_all", action="store_false",
            help="Set the multiplicative values of lr/decay to 0. Use when you don't want to finetune your pretrained weights")
    parser.add_option("-m", "--mean", dest="mean", 
       help="Mean file, usually .binaryprototxt or .npy", metavar="PATH")
    parser.add_option("-s", "--scale", dest="scale", type="float", default=1.0, 
            help="Scale param. For example, if you want to provide a way to transform your images from [0,255] to [0,1 range]", metavar="FLOAT")

    group = OptionGroup(parser, "LMDB Options",
            "How to provide the paths to your LMDB databases.")
    group.add_option("-T", "--train-lmdb", dest="train_lmdb",
            help="LMDB with train data", metavar="PATH")
    group.add_option("-L", "--train-labels-lmdb", dest="train_labels_lmdb", 
            help="LMDB with train labels", metavar="PATH")
    group.add_option("-t", "--test-lmdb", dest="test_lmdb",
            help="LMDB with test data", metavar="PATH")
    group.add_option("-l", "--test-labels-lmdb", dest="test_labels_lmdb", 
            help="LMDB with test labels", metavar="PATH")

    parser.add_option_group(group)

    (options, args) = parser.parse_args()

    alex = alexnet(train_lmdb=options.train_lmdb,
            train_labels_lmdb=options.train_labels_lmdb,
            test_lmdb=options.test_lmdb,
            test_labels_lmdb=options.test_labels_lmdb,
            batch_size=options.batch_size,
            scale=options.scale,
            train=options.train,
            num_classes=options.num_classes,
            learn_all=options.no_train_all) 

    # write the net to file
    with open('alexnet.prototxt', 'w') as f:    
        f.write(str(alex.to_proto()))
