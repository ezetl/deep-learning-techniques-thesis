import caffe
from caffe import (layers as L, params as P)

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
        raise LayerWrapperException("You forgot to provide a path to a LMDB database")

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
    conv1, relu1 = conv_relu(data0, 11, 96, stride=4, param=[weight_param('conv1_w', learn_all=learn_all), bias_param('conv1_b', learn_all=learn_all)])
    conv1_p, relu1_p = conv_relu(data1, 11, 96, stride=4, param=[weight_param('conv1_w', learn_all=learn_all), bias_param('conv1_b', learn_all=learn_all)])

    pool1 = L.Pooling(relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    pool1_p = L.Pooling(relu1_p, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    norm1 = L.LRN(pool1, local_size=5, alpha=1e-4, beta=0.75)
    norm1_p = L.LRN(pool1_p, local_size=5, alpha=1e-4, beta=0.75)

    conv2, relu2 = conv_relu(norm1, 5, 256, pad=2, group=2, param=[weight_param('conv2_w', learn_all=learn_all), bias_param('conv2_b', learn_all=learn_all)])
    conv2_p, relu2_p = conv_relu(norm1_p, 5, 256, pad=2, group=2, param=[weight_param('conv2_w', learn_all=learn_all), bias_param('conv2_b', learn_all=learn_all)])

    pool2 = L.Pooling(relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    pool2_p = L.Pooling(relu2_p, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    norm2 = L.LRN(pool2, local_size=5, alpha=1e-4, beta=0.75)
    norm2_p = L.LRN(pool2_p, local_size=5, alpha=1e-4, beta=0.75)

    if not is_mnist:
        conv3, relu3 = conv_relu(norm2, 3, 384, pad=1, param=[weight_param('conv3_w', learn_all=learn_all), bias_param('conv3_b', learn_all=learn_all)])
        conv3_p, relu3_p = conv_relu(norm2_p, 3, 384, pad=1, param=[weight_param('conv3_w', learn_all=learn_all), bias_param('conv3_b', learn_all=learn_all)])
    
        conv4, relu4 = conv_relu(relu3, 3, 384, pad=1, group=2, param=[weight_param('conv4_w', learn_all=learn_all), bias_param('conv4_b', learn_all=learn_all)])
        conv4_p, relu4_p = conv_relu(relu3_p, 3, 384, pad=1, group=2, param=[weight_param('conv4_w', learn_all=learn_all), bias_param('conv4_b', learn_all=learn_all)])
    
        conv5, relu5 = conv_relu(relu4, 3, 256, pad=1, group=2, param=[weight_param('conv5_w', learn_all=learn_all), bias_param('conv5_b', learn_all=learn_all)])
        conv5_p, relu5_p = conv_relu(relu4_p, 3, 256, pad=1, group=2, param=[weight_param('conv5_w', learn_all=learn_all), bias_param('conv5_b', learn_all=learn_all)])
    
        pool5 = L.Pooling(relu5, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        pool5_p =  L.Pooling(relu5_p, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        return pool5, pool5_p

    return norm2, norm2_p
