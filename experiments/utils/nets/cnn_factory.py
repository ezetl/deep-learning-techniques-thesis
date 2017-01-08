#!/usr/bin/env python2.7
import caffe
from caffe import (layers as L, params as P)
from layers_wrappers import *


caffe.set_device(0)
caffe.set_mode_gpu()


class MNISTNetFactory:

    @staticmethod
    def standar(lmdb_path=None, batch_size=125, scale=1.0, is_train=True, learn_all=True):
        """
        Creates a protoxt similar to the first layers of AlexNet architecture for the MNIST experiment
    
        :param lmdb_path: str. Path to train LMDB
        :param batch_size: int. Batch size
        :param scale: float. How to scale the images
        :param is_train: bool. Flag indicating if this is for testing or training
        :returns: Caffe NetSpec, tuple with names of loss blobs, tuple with name of accuracy blobs
        """
        n = caffe.NetSpec()
    
        phase = caffe.TRAIN if is_train else caffe.TEST
        n.data, n.label = L.Data(include=dict(phase=phase), batch_size=batch_size, backend=P.Data.LMDB, source=lmdb_path, transform_param=dict(scale=scale), ntop=2)
        
        n.conv1 = L.Convolution(n.data, kernel_size=11, stride=4, num_output=96, param=[weight_param('conv1_w', learn_all=learn_all), bias_param('conv1_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler)
        n.relu1 = L.ReLU(n.conv1, in_place=True)
        n.pool1 = L.Pooling(n.relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)

        n.conv2 = L.Convolution(n.norm1, kernel_size=5, num_output=256, pad=2, group=2, param=[weight_param('conv2_w', learn_all=learn_all), bias_param('conv2_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler)
        n.relu2 = L.ReLU(n.conv2, in_place=True)

        n.pool2 = L.Pooling(n.relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)

        n.fc500 = L.InnerProduct(n.norm2, num_output=500, param=[weight_param('fc500_w', learn_all=True), bias_param('fc500_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
        n.relu3 = L.ReLU(n.fc500, in_place=True)

        if is_train:
            n.dropout = fc10input = L.Dropout(n.relu3, in_place=True)
        else:
            fc10input = n.relu3
        # Learn all true because we always want to train the top classifier no matter if we are training from scratch or finetuning    
        n.fc10 = L.InnerProduct(fc10input, num_output=10, param=[weight_param('fc10_w', learn_all=True), bias_param('fc10_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
    
        if is_train:
            n.loss = L.SoftmaxWithLoss(n.fc10, n.label)
        n.acc = L.Accuracy(n.fc10, n.label, include=dict(phase=caffe.TEST))
    
        # Returning the name of the loss/acc layers is useful because then we can 
        # know which outputs of the net we can track to test the 'health' 
        # of the training process
        return n, ('loss',), ('acc',)
    
    @staticmethod
    def siamese_egomotion(lmdb_path=None, labels_lmdb_path=None,
            batch_size=125, scale=1.0, is_train=True, learn_all=False, sfa=False):
        """
        Creates a protoxt for the AlexNet architecture for the MNIST experiment
        Uses Egomotion as stated in the paper
    
        :param lmdb_path: str. Path to train LMDB
        :param labels_lmdb_path: str. Path to train LMDB labels
        :param batch_size: int. Batch size
        :param scale: float. How to scale the images
        :param is_train: bool. Flag indicating if this is for testing or training
        :param learn_all: bool. Flag indicating if we should learn all the layers from scratch
        :returns: Caffe NetSpec, tuple with names of loss blobs, tuple with name of accuracy blobs
        """
    
        n = caffe.NetSpec()
    
        n.data, n.label = input_layers(lmdb_path=lmdb_path, labels_lmdb_path=labels_lmdb_path, batch_size=batch_size, scale=scale, is_train=is_train)
        
        # Slice data/labels for MNIST
        n.data0, n.data1 = L.Slice(n.data, slice_param=dict(axis=1, slice_point=1), ntop=2)
        n.labelx, n.labely, n.labelz = L.Slice(n.label, slice_param=dict(axis=1, slice_point=[1,2]), ntop=3)
    
        # BCNN
        n.norm2, n.norm2_p = bcnn(n.data0, n.data1, n, learn_all, True)
    
        # TCNN
        n.concat = L.Concat(n.norm2, n.norm2_p, concat_param=dict(axis=1))
        n.fc1000 = L.InnerProduct(n.concat, num_output=1000, param=[weight_param('fc1000_w', learn_all=True), bias_param('fc1000_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
        n.relu3 = L.ReLU(n.fc1000, in_place=True)
    
        if is_train:
            n.dropout = fcxinput = fcyinput = fczinput = L.Dropout(n.relu3, in_place=True)
        else:
            fcxinput = fcyinput = fczinput = n.relu3
    
        # Classifiers
        n.fcx = L.InnerProduct(fcxinput, num_output=7, param=[weight_param('fcx_w', learn_all=True), bias_param('fcx_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
        n.fcy = L.InnerProduct(fcyinput, num_output=7, param=[weight_param('fcy_w', learn_all=True), bias_param('fcy_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
        n.fcz = L.InnerProduct(fczinput, num_output=20, param=[weight_param('fcz_w', learn_all=True), bias_param('fcz_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)

        n.loss_x = L.SoftmaxWithLoss(n.fcx, n.labelx)
        n.loss_y = L.SoftmaxWithLoss(n.fcy, n.labely)
        n.loss_z = L.SoftmaxWithLoss(n.fcz, n.labelz)
        n.acc_x = L.Accuracy(n.fcx, n.labelx, include=dict(phase=caffe.TEST))
        n.acc_y = L.Accuracy(n.fcy, n.labely, include=dict(phase=caffe.TEST))
        n.acc_z = L.Accuracy(n.fcz, n.labelz, include=dict(phase=caffe.TEST))
    
        return n, ('loss_x', 'loss_y', 'loss_z'), ('acc_x', 'acc_y', 'acc_z')

    @staticmethod
    def siamese_contrastive(lmdb_path=None, labels_lmdb_path=None,
            batch_size=125, scale=1.0, contrastive_margin=10, is_train=True, learn_all=False, sfa=False):
        """
        Creates a protoxt for the AlexNet architecture for the MNIST experiment
        Uses Contrastive loss
    
        :param lmdb_path: str. Path to train LMDB
        :param labels_lmdb_path: str. Path to train LMDB labels
        :param batch_size: int. Batch size
        :param scale: float. How to scale the images
        :param contrastive_margin: int. Margin for the contrastive loss layer
        :param is_train: bool. Flag indicating if this is for testing or training
        :param learn_all: bool. Flag indicating if we should learn all the layers from scratch
        :returns: Caffe NetSpec, tuple with names of loss blobs, tuple with name of accuracy blobs
        """
    
        n = caffe.NetSpec()
    
        n.data, n.label = input_layers(lmdb_path=lmdb_path, labels_lmdb_path=labels_lmdb_path, batch_size=batch_size, scale=scale, is_train=is_train)
        
        # Slice data/labels for MNIST
        n.data0, n.data1 = L.Slice(n.data, slice_param=dict(axis=1, slice_point=1), ntop=2)
    
        # BCNN
        n.norm2, n.norm2_p = bcnn(n.data0, n.data1, n, learn_all, True)
    
        # TCNNs
        n.fc1 = L.InnerProduct(n.norm2, num_output=500, param=[weight_param('fc1_p_w', learn_all=True), bias_param('fc1_p_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
        n.relu3 = L.ReLU(n.fc1, in_place=True)
        n.dropout1 = L.Dropout(n.relu3, in_place=True)
        n.fc2 = L.InnerProduct(n.relu3, num_output=100, param=[weight_param('fc2_p_w', learn_all=True), bias_param('fc2_p_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
    
        n.fc1_p = L.InnerProduct(n.norm2_p, num_output=500, param=[weight_param('fc1_p_w', learn_all=True), bias_param('fc1_p_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
        n.relu3_p = L.ReLU(n.fc1_p, in_place=True)
        n.dropout1_p = L.Dropout(n.relu3_p, in_place=True)
        n.fc2_p = L.InnerProduct(n.relu3_p, num_output=100, param=[weight_param('fc2_p_w', learn_all=True), bias_param('fc2_p_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)

        n.contrastive = L.ContrastiveLoss(n.fc2, n.fc2_p, n.label, contrastive_loss_param=dict(margin=contrastive_margin))
    
        return n, ('contrastive',), None 


class KITTINetFactory:

    @staticmethod
    def siamese_egomotion(lmdb_path=None, labels_lmdb_path=None,
            batch_size=125, scale=1.0, is_train=True, learn_all=True):
        """
        Creates a protoxt for the AlexNet architecture
    
        :param lmdb_path: str. Path to train LMDB
        :param labels_lmdb_path: str. Path to train LMDB labels
        :param test_lmdb: str. Path to train LMDB
        :param test_labels_lmdb: str. Path to test LMDB labels
        :param batch_size: int. Batch size
        :param scale: float. How to scale the images
        :param is_train: bool. Flag indicating if this is for testing or training
        :param learn_all: bool. Flag indicating if we should learn all the layers from scratch
        :returns: Caffe NetSpec, tuple with names of loss blobs, tuple with name of accuracy blobs
        """
        n = caffe.NetSpec()
    
        n.data, n.label = input_layers(lmdb_path=lmdb_path, labels_lmdb_path=labels_lmdb_path, batch_size=batch_size, scale=scale, is_train=is_train)
        
        # Slice data/labels
        n.data0, n.data1 = L.Slice(n.data, slice_param=dict(axis=1, slice_point=3), ntop=2)
        n.labelx, n.labely, n.labelz = L.Slice(n.label, slice_param=dict(axis=1, slice_point=[1,2]), ntop=3)
    
        # BCNN
        pool5, pool5_p = bcnn(n.data0, n.data1, n, learn_all, False)
    
        # TCNN
        n.concat = L.Concat(pool5, pool5_p, concat_param=dict(axis=1))


        n.conv6 = L.Convolution(n.concat, kernel_size=3, stride=2, num_output=256, pad=1, group=2, param=[weight_param('conv6_w', learn_all=learn_all), bias_param('conv6_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler)
        n.relu6 = L.ReLU(n.conv6, in_place=True)

        n.conv7 = L.Convolution(n.relu6, kernel_size=3, stride=2, num_output=128, param=[weight_param('conv7_w', learn_all=learn_all), bias_param('conv7_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler)
        n.relu7 = L.ReLU(n.conv7, in_place=True)
    
        n.fc7_ego = L.InnerProduct(n.relu7, num_output=500, param=[weight_param('fc7_ego_w', learn_all=True), bias_param('fc7_ego_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
        n.relu8 = L.ReLU(n.fc7_ego, in_place=True)
        if is_train:
            n.drop = fcxinput = fcyinput = fczinput = L.Dropout(n.relu8, in_place=True)
        else:
            fcxinput = fcyinput = fczinput = n.relu8
    
        # Classifiers
        n.fcx = L.InnerProduct(fcxinput, num_output=20, param=[weight_param('fcx_w', learn_all=True), bias_param('fcx_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
        n.fcy = L.InnerProduct(fcyinput, num_output=20, param=[weight_param('fcy_w', learn_all=True), bias_param('fcy_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
        n.fcz = L.InnerProduct(fczinput, num_output=20, param=[weight_param('fcz_w', learn_all=True), bias_param('fcz_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
    
        if is_train:
            n.loss_x = L.SoftmaxWithLoss(n.fcx, n.labelx)
            n.loss_y = L.SoftmaxWithLoss(n.fcy, n.labely)
            n.loss_z = L.SoftmaxWithLoss(n.fcz, n.labelz)
        n.acc_x = L.Accuracy(n.fcx, n.labelx, include=dict(phase=caffe.TEST))
        n.acc_y = L.Accuracy(n.fcy, n.labely, include=dict(phase=caffe.TEST))
        n.acc_z = L.Accuracy(n.fcz, n.labelz, include=dict(phase=caffe.TEST))
    
        return n, ('loss_x', 'loss_y', 'loss_z'), ('acc_x', 'acc_y', 'acc_z')
    
    @staticmethod
    def siamese_contrastive(lmdb_path=None, labels_lmdb_path=None,
            batch_size=125, scale=1.0, contrastive_margin=10, is_train=True, learn_all=True):
        """
        Creates a protoxt for siamese AlexNet architecture with a contrastive loss layer on top
    
        :param lmdb_path: str. Path to train LMDB
        :param labels_lmdb_path: str. Path to train LMDB labels
        :param test_lmdb: str. Path to train LMDB
        :param test_labels_lmdb: str. Path to test LMDB labels
        :param batch_size: int. Batch size
        :param scale: float. How to scale the images
        :param contrastive_margin: int. Margin for the contrastive loss layer
        :param is_train: bool. Flag indicating if this is for testing or training
        :param learn_all: bool. Flag indicating if we should learn all the layers from scratch
        :returns: Caffe NetSpec, tuple with names of loss blobs, tuple with name of accuracy blobs
        """
        n = caffe.NetSpec()
    
        n.data, n.label = input_layers(lmdb_path=lmdb_path, labels_lmdb_path=labels_lmdb_path, batch_size=batch_size, scale=scale, is_train=is_train)
        
        # Slice data/labels
        n.data0, n.data1 = L.Slice(n.data, slice_param=dict(axis=1, slice_point=3), ntop=2)
    
        # BCNN
        pool5, pool5_p = bcnn(n.data0, n.data1, n, learn_all, False)

        # TCNNs
        n.fc1 = L.InnerProduct(pool5, num_output=500, param=[weight_param('fc1_p_w', learn_all=True), bias_param('fc1_p_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
        n.relu3 = L.ReLU(n.fc1, in_place=True)
        n.dropout1 = L.Dropout(n.relu3, in_place=True)
        n.fc2 = L.InnerProduct(n.relu3, num_output=100, param=[weight_param('fc2_p_w', learn_all=True), bias_param('fc2_p_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
    
        n.fc1_p = L.InnerProduct(pool5_p, num_output=500, param=[weight_param('fc1_p_w', learn_all=True), bias_param('fc1_p_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
        n.relu3_p = L.ReLU(n.fc1_p, in_place=True)
        n.dropout1_p = L.Dropout(n.relu3_p, in_place=True)
        n.fc2_p = L.InnerProduct(n.relu3_p, num_output=100, param=[weight_param('fc2_p_w', learn_all=True), bias_param('fc2_p_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)

        n.contrastive = L.ContrastiveLoss(n.fc2, n.fc2_p, n.label, contrastive_loss_param=dict(margin=contrastive_margin))
    
        return n, ('contrastive',), None 
    
    @staticmethod
    def standar(lmdb_path=None, labels_lmdb_path=None, batch_size=125,
            scale=1.0, is_train=True, num_classes=397, learn_all=True, layers='5'):
        """
        Creates a protoxt for the AlexNet architecture
    
        :param lmdb_path: str. Path to train LMDB
        :param labels_lmdb_path: str. Path to train LMDB labels
        :param test_lmdb: str. Path to train LMDB
        :param test_labels_lmdb: str. Path to test LMDB labels
        :param batch_size: int. Batch size
        :param scale: float. How to scale the images
        :param is_train: bool. Flag indicating if this is for testing or training
        :param num_classes: int. number of classes for the top classifier
        :param classifier_name: str. name of the top classifier
        :param learn_all: bool. Flag indicating if we should learn all the layers from scratch
        :param layers: str. from which layer we will extract features to train a classifier
        :returns: Caffe NetSpec, tuple with names of loss blobs, tuple with name of accuracy blobs
        """
        n = caffe.NetSpec()

        n.data, n.label = input_layers(lmdb_path=lmdb_path, labels_lmdb_path=labels_lmdb_path, batch_size=batch_size, scale=scale, is_train=is_train)

        n.conv1 = L.Convolution(n.data, kernel_size=11, stride=4, num_output=96, param=[weight_param('conv1_w', learn_all=learn_all), bias_param('conv1_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler)
        n.relu1 = L.ReLU(n.conv1, in_place=True)
        n.pool1 = L.Pooling(n.relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)

        if layers == '1':
            n.fc = L.InnerProduct(n.norm1, num_output=num_classes, param=[weight_param('fc_w', learn_all=True), bias_param('fc_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
            if is_train:
                n.loss = L.SoftmaxWithLoss(n.fc, n.label)
            n.acc = L.Accuracy(n.fc, n.label, include=dict(phase=caffe.TEST))

            return n, ('loss',), ('acc',)


        n.conv2 = L.Convolution(n.norm1, kernel_size=5, num_output=256, pad=2, group=2, param=[weight_param('conv2_w', learn_all=learn_all), bias_param('conv2_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler)
        n.relu2 = L.ReLU(n.conv2, in_place=True)
        n.pool2 = L.Pooling(n.relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
        n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)

        if layers == '2':
            n.fc = L.InnerProduct(n.norm2, num_output=num_classes, param=[weight_param('fc_w', learn_all=True), bias_param('fc_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
            if is_train:
                n.loss = L.SoftmaxWithLoss(n.fc, n.label)
            n.acc = L.Accuracy(n.fc, n.label, include=dict(phase=caffe.TEST))
            return n, ('loss',), ('acc',)

        n.conv3 = L.Convolution(n.norm2, kernel_size=3, num_output=384, pad=1, param=[weight_param('conv3_w', learn_all=learn_all), bias_param('conv3_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler)
        n.relu3 = L.ReLU(n.conv3, in_place=True)

        if layers == '3':
            n.fc = L.InnerProduct(n.relu3, num_output=num_classes, param=[weight_param('fc_w', learn_all=True), bias_param('fc_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
            if is_train:
                n.loss = L.SoftmaxWithLoss(n.fc, n.label)
            n.acc = L.Accuracy(n.fc, n.label, include=dict(phase=caffe.TEST))
            return n, ('loss',), ('acc',)

        n.conv4 = L.Convolution(n.relu3, kernel_size=3, num_output=384, pad=1, group=2, param=[weight_param('conv4_w', learn_all=learn_all), bias_param('conv4_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler)
        n.relu4 = L.ReLU(n.conv4, in_place=True)

        if layers == '4':
            n.fc = L.InnerProduct(n.relu4, num_output=num_classes, param=[weight_param('fc_w', learn_all=True), bias_param('fc_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
            if is_train:
                n.loss = L.SoftmaxWithLoss(n.fc, n.label)
            n.acc = L.Accuracy(n.fc, n.label, include=dict(phase=caffe.TEST))
            return n, ('loss',), ('acc',)
        n.conv5 = L.Convolution(n.relu4, kernel_size=3, num_output=256, pad=1, group=2, param=[weight_param('conv5_w', learn_all=learn_all), bias_param('conv5_b', learn_all=learn_all)], weight_filler=weight_filler, bias_filler=bias_filler)
        n.relu5 = L.ReLU(n.conv5, in_place=True)
        n.pool5 = L.Pooling(n.relu5, pool=P.Pooling.MAX, kernel_size=3, stride=2)

        if layers == '5':
            n.fc = L.InnerProduct(n.pool5, num_output=num_classes, param=[weight_param('fc_w', learn_all=True), bias_param('fc_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
            if is_train:
                n.loss = L.SoftmaxWithLoss(n.fc, n.label)
            n.acc = L.Accuracy(n.fc, n.label, include=dict(phase=caffe.TEST))
            return n, ('loss',), ('acc',)

        n.fc6 = L.InnerProduct(n.pool5, num_output=4096, param=[weight_param('fc6_w', learn_all=True), bias_param('fc6_b', learn_all=learn_all)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
        n.relu6 = L.ReLU(n.fc6, in_place=True)

        if is_train:
            n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
        else:
            fc7input = n.relu6

        n.fc7 = L.InnerProduct(fc7input, num_output=4096, param=[weight_param('fc7_w', learn_all=True), bias_param('fc7_b', learn_all=True)], weight_filler=weight_filler_fc, bias_filler=bias_filler)
        n.relu7 = L.ReLU(n.fc7, in_place=True)
    
        if is_train:
            n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
        else:
            fc8input = n.relu7

        n.fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=[weight_param('fc8_w', learn_all=True), bias_param('fc8_b', learn_all=True)])

        if is_train:
            n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
        n.acc = L.Accuracy(n.fc8, n.label, include=dict(phase=caffe.TEST))

        return n, ('loss',), ('acc',)
