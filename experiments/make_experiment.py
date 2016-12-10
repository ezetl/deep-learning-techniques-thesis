#!/usr/bin/env python2.7
import caffe
from caffe import layers as L
from caffe import params as P
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


def alexnet(train_data, train_labels=None, test_data=None, test_labels=None,
        label=None, train=True, num_classes=1000, classifier_name='fc', 
        learn_all=False, output_proto='alexnet.prototxt'):
    """
    Creates a protoxt for the AlexNet architecture

    :param data: A list of Caffe Data layers with proper shape.
    :param label:
    :param train: bool. Flag indicating if this is for deploy or training
    :param num_classes: int. number of classes for the top classifier
    :param classifier_name: str. name of the top classifier
    :param learn_all: bool. Flag indicating if we should learn all the layers from scratch
    :returns: Caffe NetSpec 
    """
    n = caffe.NetSpec()
    n.data = train_data
    if train_labels:
        n.labels = train_labels
    if test_data:
        n.test_data = test_data
    if test_labels:
        n.test_labels = test_labels

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
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # write the net to file
    with open(output_proto, 'w') as f:    
        f.write(str(n.to_proto()))
    return n    


if __name__ == "__main__":
    dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))
    alexnet(dummy_data, train=True, learn_all=True, output_proto='alexnet.prototxt')
