#!/usr/bin/env python2.7
import caffe
from caffe import (layers as L, params as P)

caffe.set_device(0)
caffe.set_mode_gpu()


class CNNFactoryError(Exception):
    pass


def get_weight_param(name, learn_all=True):
    """
    Creates a named param for weights of a conv/fc layer

    Example of named param for weights:
    param {
      name: "conv1_w"
      lr_mult: 1
      decay_mult: 1
    }

    :param name: str
    :param learn_all: bool. If True, sets the weights of that layer to be modified during the training process
    :returns: dict with params
    """
    lr_mult = decay_mult = 1 if learn_all else 0
    return dict(name=name, lr_mult=lr_mult, decay_mult=decay_mult)


def get_bias_param(name, learn_all=True):
    """
    Creates a named param for bias of a conv/fc layer

    Example of named param for bias:
    param {
      name: "conv1_b"
      lr_mult: 1
      decay_mult: 1
    }

    :param name: str
    :param learn_all: bool. If True, sets the weights of that layer to be modified during the training process
    :returns: dict with params
    """
    lr_mult = 2 if learn_all else 0
    return dict(name=name, lr_mult=lr_mult, decay_mult=0)


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
    :param param: 
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


def input_layers(lmdb_path=None, labels_lmdb_path=None, batch_size=125,
        scale=1.0, is_train=True):
    """
    Creates the Data and Slice layers needed for the experiments with siamese networks 

    :param lmdb_path: str. Path to train LMDB
    :param labels_lmdb_path: str. Path to train LMDB labels
    :param batch_size: int. Batch size
    :param scale: float. How to scale the images
    :param is_train: bool. Flag indicating if this is for deploy/testing or training
    :returns: data and label Caffe layers 
    """
    phase = caffe.TRAIN if is_train else caffe.TEST
    if lmdb_path and labels_lmdb_path:
        data = L.Data(include=dict(phase=phase), batch_size=batch_size, backend=P.Data.LMDB, source=lmdb_path, transform_param=dict(scale=scale), ntop=1)
        label = L.Data(include=dict(phase=phase), batch_size=batch_size, backend=P.Data.LMDB, source=labels_lmdb_path, ntop=1)
    elif lmdb_path and not labels_lmdb_path:
        data, label = L.Data(batch_size=batch_size, include=dict(phase=phase), backend=P.Data.LMDB, source=lmdb_path, transform_param=dict(scale=scale), ntop=2)
    else:
        raise CNNFactoryError("You forgot to provide a path to a LMDB database")

    return data, label 


def bcnn(data0, data1, learn_all, is_mnist):
    """
    Creates the Base Convolutional Network from the paper 
    "Learning To See By Moving" by Agrawal et al.

    :param data: L.Data layer from Caffe
    :param learn_all: bool. Flag indicating if this bccn is to train from scratch or to finetune 
    :param mnist: bool. Flag indicating if this bcnn is for mnist (just use 2 conv layers or if it is for kitti/sf/etc (use all conv layers) 
    :returns the last layers of this BCNN (L.LRN)
    """
    conv1, relu1 = conv_relu(data0, 11, 96, stride=4, param=[get_weight_param('conv1_w', learn_all=learn_all), get_bias_param('conv1_b', learn_all=learn_all)])
    conv1_p, relu1_p = conv_relu(data1, 11, 96, stride=4, param=[get_weight_param('conv1_w', learn_all=learn_all), get_bias_param('conv1_b', learn_all=learn_all)])

    pool1 = L.Pooling(relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    pool1_p = L.Pooling(relu1_p, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    norm1 = L.LRN(pool1, local_size=5, alpha=1e-4, beta=0.75)
    norm1_p = L.LRN(pool1_p, local_size=5, alpha=1e-4, beta=0.75)

    conv2, relu2 = conv_relu(norm1, 5, 256, pad=2, group=2, param=[get_weight_param('conv2_w', learn_all=learn_all), get_bias_param('conv2_b', learn_all=learn_all)])
    conv2_p, relu2_p = conv_relu(norm1_p, 5, 256, pad=2, group=2, param=[get_weight_param('conv2_w', learn_all=learn_all), get_bias_param('conv2_b', learn_all=learn_all)])

    pool2 = L.Pooling(relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    pool2_p = L.Pooling(relu2_p, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    norm2 = L.LRN(pool2, local_size=5, alpha=1e-4, beta=0.75)
    norm2_p = L.LRN(pool2_p, local_size=5, alpha=1e-4, beta=0.75)

    if not is_mnist:
        conv3, relu3 = conv_relu(norm2, 3, 384, pad=1, param=[get_weight_param('conv3_w', learn_all=learn_all), get_bias_param('conv3_b', learn_all=learn_all)])
        conv3_p, relu3_p = conv_relu(norm2_p, 3, 384, pad=1, param=[get_weight_param('conv3_w', learn_all=learn_all), get_bias_param('conv3_b', learn_all=learn_all)])
    
        conv4, relu4 = conv_relu(relu3, 3, 384, pad=1, group=2, param=[get_weight_param('conv4_w', learn_all=learn_all), get_bias_param('conv4_b', learn_all=learn_all)])
        conv4_p, relu4_p = conv_relu(relu3_p, 3, 384, pad=1, group=2, param=[get_weight_param('conv4_w', learn_all=learn_all), get_bias_param('conv4_b', learn_all=learn_all)])
    
        conv5, relu5 = conv_relu(relu4, 3, 256, pad=1, group=2, param=[get_weight_param('conv5_w', learn_all=learn_all), get_bias_param('conv5_b', learn_all=learn_all)])
        conv5_p, relu5_p = conv_relu(relu4_p, 3, 256, pad=1, group=2, param=[get_weight_param('conv5_w', learn_all=learn_all), get_bias_param('conv5_b', learn_all=learn_all)])
    
        pool5 = L.Pooling(relu5, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        pool5_p =  L.Pooling(relu5_p, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        return pool5, pool5_p

    return norm2, norm2_p


def standard_alexnet_mnist(lmdb_path=None, batch_size=125, scale=1.0, is_train=True, learn_all=True):
    """
    Creates a protoxt similar to the first layers of AlexNet architecture for the MNIST experiment

    :param lmdb_path: str. Path to train LMDB
    :param batch_size: int. Batch size
    :param scale: float. How to scale the images
    :param is_train: bool. Flag indicating if this is for deploy/testing or training
    :returns: Caffe NetSpec
    """

    n = caffe.NetSpec()

    phase = caffe.TRAIN if is_train else caffe.TEST
    n.data, n.label = L.Data(include=dict(phase=phase), batch_size=batch_size, backend=P.Data.LMDB, source=lmdb_path, transform_param=dict(scale=scale), ntop=2)
    
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=[get_weight_param('conv1_w', learn_all=learn_all), get_bias_param('conv1_b', learn_all=learn_all)])
    n.pool1 = L.Pooling(n.relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=[get_weight_param('conv2_w', learn_all=learn_all), get_bias_param('conv2_b', learn_all=learn_all)])
    n.pool2 = L.Pooling(n.relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.fc, n.relu3 = fc_relu(n.norm2, 500, param=[get_weight_param('fc_w', learn_all=learn_all), get_bias_param('fc_b', learn_all=learn_all)])
    if is_train:
        n.drop = fc10input = L.Dropout(n.relu3, in_place=True)
    else:
        fc10input = n.relu3
    # Train true because we always want to train the top classifier no matter if we are training from scratch or finetuning    
    n.fc10 = fc(fc10input, 10, param=[get_weight_param('fc10_w', lean_all=True), get_bias_param('fc10_b', learn_all=True)])

    if not is_train:
        n.probs = L.Softmax(n.fc10)
    else:
        n.loss = L.SoftmaxWithLoss(n.fc10, n.labelx)
        n.acc = L.Accuracy(n.fc10, n.label, include=dict(phase=caffe.TEST))

    return n


def siamese_alexnet_mnist(lmdb_path=None, labels_lmdb_path=None,
        batch_size=125, scale=1.0, is_train=True, learn_all=False, sfa=False):
    """
    Creates a protoxt for the AlexNet architecture for the MNIST experiment

    :param lmdb_path: str. Path to train LMDB
    :param labels_lmdb_path: str. Path to train LMDB labels
    :param batch_size: int. Batch size
    :param scale: float. How to scale the images
    :param is_train: bool. Flag indicating if this is for deploy/testing or training
    :param learn_all: bool. Flag indicating if we should learn all the layers from scratch
    :returns: Caffe NetSpec
    """

    n = caffe.NetSpec()

    n.data, n.label = input_layers(lmdb_path=lmdb_path, labels_lmdb_path=labels_lmdb_path, batch_size=batch_size, scale=scale, is_train=is_train)
    
    # Slice data/labels for MNIST
    n.data0, n.data1 = L.Slice(n.data, slice_param=dict(axis=1, slice_point=1), ntop=2)
    n.labelx, n.labely, n.labelz = L.Slice(n.label, slice_param=dict(axis=1, slice_point=[1,2]), ntop=3)

    # BCNN
    n.norm2, n.norm2_p = bcnn(n.data0, n.data1, learn_all, True)

    # TCNN
    n.concat = L.Concat(n.norm2, n.norm2_p, concat_param=dict(axis=1))
    n.fc, n.relu3 = fc_relu(n.concat, 1000, param=[get_weight_param('fc_w', learn_all=learn_all), get_bias_param('fc_b', learn_all=learn_all)])

    if is_train:
        n.drop = fcxinput = fcyinput = fczinput = L.Dropout(n.relu3, in_place=True)
    else:
        fcxinput = fcyinput = fczinput = n.relu3

    # Classifiers
    n.fcx = fc(fcxinput, 7, param=[get_weight_param('fcx_w', learn_all=learn_all), get_bias_param('fcx_b', learn_all=learn_all)])
    n.fcy = fc(fcyinput, 7, param=[get_weight_param('fcy_w', learn_all=learn_all), get_bias_param('fcy_b', learn_all=learn_all)])
    n.fcz = fc(fczinput, 20, param=[get_weight_param('fcz_w', learn_all=learn_all), get_bias_param('fcz_b', learn_all=learn_all)])

    if not is_train:
        n.probsx = L.Softmax(n.fcx)
        n.probsy = L.Softmax(n.fcy)
        n.probsz = L.Softmax(n.fcz)
    else:
        n.loss_x = L.SoftmaxWithLoss(n.fcx, n.labelx)
        n.loss_y = L.SoftmaxWithLoss(n.fcy, n.labely)
        n.loss_z = L.SoftmaxWithLoss(n.fcz, n.labelz)
        n.acc_x = L.Accuracy(n.fcx, n.labelx, include=dict(phase=caffe.TEST))
        n.acc_y = L.Accuracy(n.fcy, n.labely, include=dict(phase=caffe.TEST))
        n.acc_z = L.Accuracy(n.fcz, n.labelz, include=dict(phase=caffe.TEST))

    return n


def siamese_alexnet_kitti(lmdb_path=None, labels_lmdb_path=None,
        batch_size=125, scale=1.0, is_train=True, learn_all=False):
    """
    Creates a protoxt for the AlexNet architecture

    :param lmdb_path: str. Path to train LMDB
    :param labels_lmdb_path: str. Path to train LMDB labels
    :param test_lmdb: str. Path to train LMDB
    :param test_labels_lmdb: str. Path to test LMDB labels
    :param batch_size: int. Batch size
    :param scale: float. How to scale the images
    :param is_train: bool. Flag indicating if this is for deploy or training
    :param learn_all: bool. Flag indicating if we should learn all the layers from scratch
    :returns: Caffe NetSpec
    """
    n = caffe.NetSpec()

    n.data, n.label = input_layers(lmdb_path=lmdb_path, labels_lmdb_path=labels_lmdb_path, batch_size=batch_size, scale=scale, is_train=is_train)
    
    # Slice data/labels
    n.data0, n.data1 = L.Slice(n.data, slice_param=dict(axis=1, slice_point=3), ntop=2)
    n.labelx, n.labely, n.labelz = L.Slice(n.label, slice_param=dict(axis=1, slice_point=[1,2]), ntop=3)

    # BCNN
    n.pool5, n.pool5_p = bcnn(n.data0, n.data1, learn_all, False)

    # TCNN
    n.concat = L.Concat(n.pool5, n.pool5_p, concat_param=dict(axis=1))
    n.conv6, n.relu6 = conv_relu(n.concat, 3, 256, stride=2, pad=1, group=2, param=[get_weight_param('conv6_w', learn_all=learn_all), get_bias_param('conv6_b', learn_all=learn_all)])
    n.conv7, n.relu7 = conv_relu(n.relu6, 3, 128, stride=2, param=[get_weight_param('conv7_w', learn_all=learn_all), get_bias_param('conv7_b', learn_all=learn_all)])

    n.fc7, n.relu8 = fc_relu(n.relu7, 500, param=[get_weight_param('fc7_w', learn_all=learn_all), get_bias_param('fc7_b', learn_all=learn_all)])
    if is_train:
        n.drop = fcxinput = fcyinput = fczinput = L.Dropout(n.relu8, in_place=True)
    else:
        fcxinput = fcyinput = fczinput = n.relu8

    # Classifiers
    n.fcx = fc(fcxinput, 20, param=[get_weight_param('fcx_w', learn_all=learn_all), get_bias_param('fcx_b', learn_all=learn_all)])
    n.fcy = fc(fcyinput, 20, param=[get_weight_param('fcy_w', learn_all=learn_all), get_bias_param('fcy_b', learn_all=learn_all)])
    n.fcz = fc(fczinput, 20, param=[get_weight_param('fcz_w', learn_all=learn_all), get_bias_param('fcz_b', learn_all=learn_all)])

    if not is_train:
        n.probsx = L.Softmax(n.fcx)
        n.probsy = L.Softmax(n.fcy)
        n.probsz = L.Softmax(n.fcz)
    else:
        n.loss_x = L.SoftmaxWithLoss(n.fcx, n.labelx)
        n.loss_y = L.SoftmaxWithLoss(n.fcy, n.labely)
        n.loss_z = L.SoftmaxWithLoss(n.fcz, n.labelz)
        n.acc_x = L.Accuracy(n.fcx, n.labelx, include=dict(phase=caffe.TEST))
        n.acc_y = L.Accuracy(n.fcy, n.labely, include=dict(phase=caffe.TEST))
        n.acc_z = L.Accuracy(n.fcz, n.labelz, include=dict(phase=caffe.TEST))

    return n


def alexnet(lmdb_path=None, labels_lmdb_path=None, batch_size=125,
        scale=1.0, is_train=True, num_classes=1000, learn_all=False):
    """
    Creates a protoxt for the AlexNet architecture

    :param lmdb_path: str. Path to train LMDB
    :param labels_lmdb_path: str. Path to train LMDB labels
    :param test_lmdb: str. Path to train LMDB
    :param test_labels_lmdb: str. Path to test LMDB labels
    :param batch_size: int. Batch size
    :param scale: float. How to scale the images
    :param is_train: bool. Flag indicating if this is for deploy or training
    :param num_classes: int. number of classes for the top classifier
    :param classifier_name: str. name of the top classifier
    :param learn_all: bool. Flag indicating if we should learn all the layers from scratch
    :returns: Caffe NetSpec
    """
    n = caffe.NetSpec()

    n.data, n.label = input_layers(lmdb_path=lmdb_path, labels_lmdb_path=labels_lmdb_path, batch_size=batch_size, scale=scale, is_train=is_train)

    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=[get_weight_param('conv1_w', learn_all=learn_all), get_bias_param('conv1_b', learn_all=learn_all)])
    n.pool1 = L.Pooling(n.relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=[get_weight_param('conv2_w', learn_all=learn_all), get_bias_param('conv2_b', learn_all=learn_all)])
    n.pool2 = L.Pooling(n.relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=[get_weight_param('conv3_w', learn_all=learn_all), get_bias_param('conv3_b', learn_all=learn_all)])
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=[get_weight_param('conv4_w', learn_all=learn_all), get_bias_param('conv4_b', learn_all=learn_all)])
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=[get_weight_param('conv5_w', learn_all=learn_all), get_bias_param('conv5_b', learn_all=learn_all)])
    n.pool5 = L.Pooling(n.relu5, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=[get_weight_param('fc6_w', learn_all=learn_all), get_bias_param('fc6_b', learn_all=learn_all)])

    if is_train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6

    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=[get_weight_param('fc7_w', learn_all=learn_all), get_bias_param('fc7_b', learn_all=learn_all)])

    if is_train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7

    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=[get_weight_param('fc8', learn_all=learn_all), get_bias_param('fc8', learn_all=learn_all)])

    if not is_train:
        n.probs = L.Softmax(fc8, include=dict(phase=phase))
    elif n.label is not None:
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label, include=dict(phase=caffe.TEST))

    return n
