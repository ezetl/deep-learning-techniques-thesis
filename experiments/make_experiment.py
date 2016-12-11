#!/usr/bin/env python2.7
from optparse import OptionParser, OptionGroup
import caffe
from caffe import (layers as L, params as P)
import numpy as np

caffe.set_device(0)
caffe.set_mode_gpu()


def get_weight_param(name, train=True):
    """
    Creates a named param for weights of a conv/fc layer

    Example of named param for weights:
    param {
      name: "conv1_w"
      lr_mult: 1
      decay_mult: 1
    }

    :param name: str
    :param train: bool. If True, sets the weights of that layer to be modified by the training process
    :returns: dict with params
    """
    if train:
        return dict(name=name, lr_mult=1, decay_mult=1)
    else:
        return dict(name=name, lr_mult=0, decay_mult=0)


def get_bias_param(name, train=True):
    """
    Creates a named param for bias of a conv/fc layer

    Example of named param for bias:
    param {
      name: "conv1_b"
      lr_mult: 1
      decay_mult: 1
    }

    :param name: str
    :param train: bool. If True, sets the weights of that layer to be modified by the training process
    :returns: dict with params
    """
    if train:
        return dict(name=name, lr_mult=2, decay_mult=0)
    else:
        return dict(name=name, lr_mult=0, decay_mult=0)


def conv_relu(bottom, ks, num_out, stride=1, pad=0, group=1,
        param={}, weight_filler=dict(type='gaussian', std=0.01),
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
    :returns: Convolution layer, ReLU layer
    """
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                num_output=num_out, pad=pad, group=group, param=param,
                weight_filler=weight_filler, bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)


def fc_relu(bottom, num_out, param={},
    weight_filler=dict(type='gaussian', std=0.005),
    bias_filler=dict(type='constant', value=0.1)):
    """
    Creates a fully connected+relu layer 

    :param bottom: Caffe layer. The layer that feed this fc.
    :param num_out: int total number of outputs
    :param param: [dict]. Multiplying factors (lr and decay) of the weights and bias.
    :param weight_filler: dict. The weight filler can be wether 'xavier' or 'gaussian' for these experiments
    :param bias_filler: dict. Usually {'constant':1}
    :returns: InnerProduct layer, ReLU layer
    """
    fc = L.InnerProduct(bottom, num_output=num_out, param=param,
            weight_filler=weight_filler, bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)


def fc(bottom, num_out, param={},
    weight_filler=dict(type='gaussian', std=0.005),
    bias_filler=dict(type='constant', value=0.1)):
    """
    Same as fc_relu, but without ReLU 

    :param bottom: Caffe layer that feeds this fc 
    :param num_out: int total number of outputs
    :returns: InnerProduct Layer
    """
    return L.InnerProduct(bottom, num_output=num_out, param=param,
            weight_filler=weight_filler, bias_filler=bias_filler)


def max_pool(bottom, ks, stride=1):
    """
    Creates a MAX Pooling layer 

    :param bottom: Caffe layer. The layer that feed this max pooling layer.
    :param ks: int size of one of the kernel sides (it is a square) 
    :param stride: int
    :returns: Caffe Pooling layer
    """
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def siamese_alexnet_kitti(train_lmdb=None, train_labels_lmdb=None,
        test_lmdb=None, test_labels_lmdb=None, batch_size=125, 
        scale=1.0, train=True, learn_all=False):
    """
    Creates a protoxt for the AlexNet architecture

    :param train_lmdb: str. Path to train LMDB
    :param train_labels_lmdb: str. Path to train LMDB labels
    :param test_lmdb: str. Path to train LMDB
    :param test_labels_lmdb: str. Path to test LMDB labels
    :param batch_size: int. Batch size
    :param scale: float. How to scale the images
    :param train: bool. Flag indicating if this is for deploy or training
    :param learn_all: bool. Flag indicating if we should learn all the layers from scratch
    :returns: Caffe NetSpec 
    """
    n = caffe.NetSpec()

    if train_lmdb and train_labels_lmdb:
        n.data = L.Data(include=dict(phase=caffe.TRAIN), batch_size=batch_size, backend=P.Data.LMDB, source=train_lmdb, transform_param=dict(scale=scale), ntop=1)
        n.label = L.Data(include=dict(phase=caffe.TRAIN), batch_size=batch_size, backend=P.Data.LMDB, source=train_labels_lmdb, ntop=1)
    elif train_lmdb and not train_labels_lmdb:
        n.data, n.label = L.Data(batch_size=batch_size, include=dict(phase=caffe.TRAIN), backend=P.Data.LMDB, source=train_lmdb, transform_param=dict(scale=scale), ntop=2)

    # TODO(eze): try to figure out a way to include train+test data in the same network by using the same blob names ('data', 'label') 
    #if test_lmdb and test_labels_lmdb:
    #    n.test_data = L.Data(include=dict(phase=caffe.TEST), batch_size=batch_size, backend=P.Data.LMDB, source=test_lmdb, transform_param=dict(scale=scale), ntop=1)
    #    n.test_label = L.Data(include=dict(phase=caffe.TEST), batch_size=batch_size, backend=P.Data.LMDB, source=test_labels_lmdb, ntop=1)
    #elif test_lmdb and not test_labels_lmdb:
    #    n.test_data, n.test_label = L.Data(batch_size=batch_size, include=dict(phase=caffe.TEST), backend=P.Data.LMDB, source=test_lmdb, transform_param=dict(scale=scale), ntop=2)

    if not train_lmdb:   
        n.data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))

    # Slice data/labels    
    n.data0, n.data1 = L.Slice(n.data, name="slice_data", slice_param=dict(axis=1, slice_point=3), ntop=2)
    n.labelx, n.labely, n.labelz = L.Slice(n.label, name="slice_labels", slice_param=dict(axis=1, slice_point=[1,2]), ntop=3)

    # BCNN
    n.conv1, n.relu1 = conv_relu(n.data0, 11, 96, stride=4, param=[get_weight_param('conv1_w', train=train), get_bias_param('conv1_b', train=train)])
    n.conv1_p, n.relu1_p = conv_relu(n.data1, 11, 96, stride=4, param=[get_weight_param('conv1_w', train=train), get_bias_param('conv1_b', train=train)])

    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.pool1_p = max_pool(n.relu1_p, 3, stride=2)

    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.norm1_p = L.LRN(n.pool1_p, local_size=5, alpha=1e-4, beta=0.75)

    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2,param=[get_weight_param('conv2_w', train=train), get_bias_param('conv2_b', train=train)])
    n.conv2_p, n.relu2_p = conv_relu(n.norm1_p, 5, 256, pad=2, group=2,param=[get_weight_param('conv2_w', train=train), get_bias_param('conv2_b', train=train)])

    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.pool2_p = max_pool(n.relu2_p, 3, stride=2)

    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.norm2_p = L.LRN(n.pool2_p, local_size=5, alpha=1e-4, beta=0.75)

    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=[get_weight_param('conv3_w', train=train), get_bias_param('conv3_b', train=train)])
    n.conv3_p, n.relu3_p = conv_relu(n.norm2_p, 3, 384, pad=1, param=[get_weight_param('conv3_w', train=train), get_bias_param('conv3_b', train=train)])

    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=[get_weight_param('conv4_w', train=train), get_bias_param('conv4_b', train=train)])
    n.conv4_p, n.relu4_p = conv_relu(n.relu3_p, 3, 384, pad=1, group=2, param=[get_weight_param('conv4_w', train=train), get_bias_param('conv4_b', train=train)])

    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=[get_weight_param('conv5_w', train=train), get_bias_param('conv5_b', train=train)])
    n.conv5_p, n.relu5_p = conv_relu(n.relu4_p, 3, 256, pad=1, group=2, param=[get_weight_param('conv5_w', train=train), get_bias_param('conv5_b', train=train)])

    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.pool5_p = max_pool(n.relu5_p, 3, stride=2)

    # TCNN
    n.concat = L.Concat(n.pool5, n.pool5_p, concat_param=dict(axis=1))
    n.conv6, n.relu6 = conv_relu(n.concat, 3, 256, stride=2, pad=1, group=2, param=[get_weight_param('conv6_w', train=train), get_bias_param('conv6_b', train=train)])
    n.conv7, n.relu7 = conv_relu(n.relu6, 3, 128, stride=2, param=[get_weight_param('conv7_w', train=train), get_bias_param('conv7_b', train=train)])

    n.fc7, n.relu8 = fc_relu(n.relu7, 500, param=[get_weight_param('fc7_w', train=train), get_bias_param('fc7_b', train=train)])
    if train:
        n.drop = fcxinput = fcyinput = fczinput = L.Dropout(n.fc7, in_place=True)
    else:
        fcxinput = n.relu7

    # Classifiers
    n.fcx = fc(fcxinput, 20, param=[get_weight_param('fcx_w', train=train), get_bias_param('fcx_b', train=train)])
    n.fcy = fc(fcyinput, 20, param=[get_weight_param('fcy_w', train=train), get_bias_param('fcy_b', train=train)])
    n.fcz = fc(fczinput, 20, param=[get_weight_param('fcz_w', train=train), get_bias_param('fcz_b', train=train)])

    if not train:
        n.probsx = L.Softmax(n.fcx)
        n.probsy = L.Softmax(n.fcy)
        n.probsz = L.Softmax(n.fcz)
    else:    
        n.lossx = L.SoftmaxWithLoss(n.fcx, n.labelx)
        n.lossy = L.SoftmaxWithLoss(n.fcy, n.labely)
        n.lossz = L.SoftmaxWithLoss(n.fcz, n.labelz)
        n.accx = L.Accuracy(n.fcx, n.labelx, include=dict(phase=caffe.TEST))
        n.accy = L.Accuracy(n.fcy, n.labely, include=dict(phase=caffe.TEST))
        n.accz = L.Accuracy(n.fcz, n.labelz, include=dict(phase=caffe.TEST))

    return n

def alexnet(train_lmdb=None, train_labels_lmdb=None, test_lmdb=None, test_labels_lmdb=None,
        batch_size=125, scale=1.0, train=True, num_classes=1000, learn_all=False):
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

    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=[get_weight_param('conv1_w', train=train), get_bias_param('conv1_b', train=train)])
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2,param=[get_weight_param('conv2_w', train=train), get_bias_param('conv2_b', train=train)])
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=[get_weight_param('conv3_w', train=train), get_bias_param('conv3_b', train=train)])
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=[get_weight_param('conv4_w', train=train), get_bias_param('conv4_b', train=train)])
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=[get_weight_param('conv5_w', train=train), get_bias_param('conv5_b', train=train)])
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=[get_weight_param('fc6_w', train=train), get_bias_param('fc6_b', train=train)])

    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6

    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=[get_weight_param('fc7_w', train=train), get_bias_param('fc7_b', train=train)])

    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7

    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=[get_weight_param('fc8', train=train), get_bias_param('fc8', train=train)])

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
    parser.add_option("-a", dest="train_all", action="store_true",
            help="Set the multiplicative values of lr/decay different from 0. Don't use if you don't want to finetune your pretrained weights")
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
    group_example = OptionGroup(parser, "Example:",
            ' '.join(['./make_experiment.py -b 125 -r -n 397\\',
                '-T /media/eze/Datasets/MNIST/mnist_train_siamese_lmdb\\', 
                '-L /media/eze/Datasets/MNIST/mnist_train_siamese_lmdb_labels\\', 
                '-t /media/eze/Datasets/MNIST/mnist_test_siamese_lmdb\\',
                '-l /media/eze/Datasets/MNIST/mnist_test_siamese_lmdb_labels\\', 
                '-s $(python -c"print(1/255.0)") -a']))

    parser.add_option_group(group)
    parser.add_option_group(group_example)

    (options, args) = parser.parse_args()

    alex = alexnet(
            train_lmdb=options.train_lmdb,
            train_labels_lmdb=options.train_labels_lmdb,
            test_lmdb=options.test_lmdb,
            test_labels_lmdb=options.test_labels_lmdb,
            batch_size=options.batch_size,
            scale=options.scale,
            train=options.train,
            num_classes=options.num_classes,
            learn_all=options.train_all
            ) 

    siam_alex = siamese_alexnet_kitti(
            train_lmdb=options.train_lmdb,
            train_labels_lmdb=options.train_labels_lmdb,
            test_lmdb=options.test_lmdb,
            test_labels_lmdb=options.test_labels_lmdb,
            batch_size=options.batch_size,
            scale=options.scale,
            train=options.train,
            learn_all=options.train_all
            ) 

    # write the nets
    with open('alexnet.prototxt', 'w') as f:    
        f.write(str(alex.to_proto()))
    with open('siamese_alexnet.prototxt', 'w') as f:    
        f.write(str(siam_alex.to_proto()))
