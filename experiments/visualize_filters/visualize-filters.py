#!/usr/bin/env python2.7
import numpy as np
import matplotlib.pyplot as plt
import sys
CAFFE_ROOT = '/home/ezetl/Software/caffe'
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe

caffe.set_mode_cpu()

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# Params
# CAFFEMODEL = '../mnist/snapshots/mnist_standar_finetuning_1000_lr0.001_iter_4000.caffemodel'
# DEPLOY = '../mnist/test/deploy/deploy_finetuning_mnist.prototxt'


#CAFFEMODEL = '../kitti/snapshots/lr0.001/snapshot_kitti_egomotion_lr0.001_iter_60000.caffemodel'
CAFFEMODEL = '/home/ezetl/Documents/lsm/kittinet_con-conv5_scratch_pad24_imS227_con-conv_iter_60000.caffemodel'
#DEPLOY = '../kitti/test/deploy/deploy.prototxt'
DEPLOY = '/home/ezetl/Documents/lsm/kitti_finetune_conv5_deploy.prototxt'
IMSIZE = 227  # 28 for mnist, 227 for kitti
IMCHANNEL = 3  # 1 for mnist, 3 for kitti
# image number 4 containing picture of a number "5"
# TEST_IM = './images/mnist_4_5.bmp'
TEST_IM = './images/kitti_000889.png' # kitti


def set_net(deploy, caffemodel, num_channels, im_size):
    net = caffe.Net(DEPLOY, CAFFEMODEL, caffe.TEST)  # test mode (e.g., don't perform dropout)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    if num_channels == 3:
        transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    net.blobs['data'].reshape(1,                 # batch size
                              num_channels,      # 3-channel (BGR) images
                              im_size, im_size)  # image size is 227x227
    return (net, transformer)


def get_conv_filters(net, conv_name):
    filters = net.params[conv_name][0].data
    s = filters.shape
    if (s[1] == 3):
        filters = np.reshape(filters, (s[0], s[2], s[3], s[1]))
    else:
        filters = np.reshape(filters, (s[0], s[2], s[3]))
    return filters


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
    and visualize each (height, width) thing in a grid of size approx.
    sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))                # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) +
                                                           tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    net, transformer = set_net(DEPLOY, CAFFEMODEL, IMCHANNEL, IMSIZE)
    # the parameters are a list of [weights, biases]
    filters = get_conv_filters(net, 'conv1')
    vis_square(filters)
