import caffe
from caffe import (layers as L, params as P)
from os.path import exists

weight_filler = dict(type='gaussian', std=0.01)
weight_filler_fc = dict(type='gaussian', std=0.005)
weight_filler_fc_xavier = dict(type='xavier')
bias_filler = dict(type='constant', value=0.1)
bias_filler_0 = dict(type='constant', value=0)
bias_filler_1 = dict(type='constant', value=1)

class LayerWrapperException(Exception):
    pass


def weight_param(name, learn_all=True):
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


def bias_param(name, learn_all=True):
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


def input_layers(lmdb_path=None, labels_lmdb_path=None, mean_file=None, batch_size=125,
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
    #transform_param=dict(scale=scale)
    transform_param=dict()
    if mean_file is not None and exists(mean_file):
        transform_param['mean_file'] = mean_file

    if lmdb_path and labels_lmdb_path:
        data = L.Data(include=dict(phase=phase), batch_size=batch_size, backend=P.Data.LMDB, source=lmdb_path, transform_param=transform_param, ntop=1)
        label = L.Data(include=dict(phase=phase), batch_size=batch_size, backend=P.Data.LMDB, source=labels_lmdb_path, ntop=1)
    elif lmdb_path and not labels_lmdb_path:
        data, label = L.Data(batch_size=batch_size, include=dict(phase=phase), backend=P.Data.LMDB, source=lmdb_path, transform_param=transform_param, ntop=2)
    else:
        raise LayerWrapperException("You forgot to provide a path to a LMDB database")

    return data, label


def bcnn(data0, data1, n, learn_all, is_mnist):
    """
    Creates the Base Convolutional Network from the paper
    "Learning To See By Moving" by Agrawal et al.

    :param data{0,1}: L.Data layers from Caffe
    :param n: Caffe NetSpec. The BCNN will be appended to this NetSpec
    :param learn_all: bool. Flag indicating if this bccn is to train from scratch or to finetune
    :param mnist: bool. Flag indicating if this bcnn is for mnist (just use 2 conv layers or if it is for kitti/sf/etc (use all conv layers)
    """
    n.conv1 = L.Convolution(n.data0, kernel_size=11, stride=4, num_output=96, param=[weight_param('conv1_w', learn_all=learn_all), bias_param('conv1_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler_0)
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.pool1 = L.Pooling(n.relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)

    n.conv1_p = L.Convolution(n.data1, kernel_size=11, stride=4, num_output=96, param=[weight_param('conv1_w', learn_all=learn_all), bias_param('conv1_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler_0)
    n.relu1_p = L.ReLU(n.conv1_p, in_place=True)
    n.pool1_p = L.Pooling(n.relu1_p, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    n.norm1_p = L.LRN(n.pool1_p, local_size=5, alpha=1e-4, beta=0.75)

    n.conv2 = L.Convolution(n.norm1, kernel_size=5, num_output=256, pad=2, group=2, param=[weight_param('conv2_w', learn_all=learn_all), bias_param('conv2_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler_0)
    n.relu2 = L.ReLU(n.conv2, in_place=True)
    n.pool2 = L.Pooling(n.relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)

    n.conv2_p = L.Convolution(n.norm1_p, kernel_size=5, num_output=256, pad=2, group=2, param=[weight_param('conv2_w', learn_all=learn_all), bias_param('conv2_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler_0)
    n.relu2_p = L.ReLU(n.conv2_p, in_place=True)
    n.pool2_p = L.Pooling(n.relu2_p, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    n.norm2_p = L.LRN(n.pool2_p, local_size=5, alpha=1e-4, beta=0.75)

    if not is_mnist:
        n.conv3 = L.Convolution(n.norm2, kernel_size=3, num_output=384, pad=1, param=[weight_param('conv3_w', learn_all=learn_all), bias_param('conv3_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler_0)
        n.relu3 = L.ReLU(n.conv3, in_place=True)

        n.conv3_p = L.Convolution(n.norm2_p, kernel_size=3, num_output=384, pad=1, param=[weight_param('conv3_w', learn_all=learn_all), bias_param('conv3_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler_0)
        n.relu3_p = L.ReLU(n.conv3_p, in_place=True)

        n.conv4 = L.Convolution(n.relu3, kernel_size=3, num_output=384, pad=1, group=2, param=[weight_param('conv4_w', learn_all=learn_all), bias_param('conv4_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler_0)
        n.relu4 = L.ReLU(n.conv4, in_place=True)

        n.conv4_p = L.Convolution(n.relu3_p, kernel_size=3, num_output=384, pad=1, group=2, param=[weight_param('conv4_w', learn_all=learn_all), bias_param('conv4_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler_0)
        n.relu4_p = L.ReLU(n.conv4_p, in_place=True)

        n.conv5 = L.Convolution(n.relu4, kernel_size=3, num_output=256, pad=1, group=2, param=[weight_param('conv5_w', learn_all=learn_all), bias_param('conv5_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler_0)
        n.relu5 = L.ReLU(n.conv5, in_place=True)

        n.conv5_p = L.Convolution(n.relu4_p, kernel_size=3, num_output=256, pad=1, group=2, param=[weight_param('conv5_w', learn_all=learn_all), bias_param('conv5_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler_0)
        n.relu5_p = L.ReLU(n.conv5_p, in_place=True)

        return n.relu5, n.relu5_p

    return n.norm2, n.norm2_p
