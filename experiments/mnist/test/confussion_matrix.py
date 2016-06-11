#!/usr/bin/env python2.7
import time
import numpy as np
import caffe
import os
import sys


classes = [str(i) for i in range(0,10)]

ROOT_IMAGES = '../data/mnist_test_standar'
IMGS_LIST = './test_mnist.txt'
DEPLOY_FILE = './deploy/deploy_finetuning_mnist.prototxt'
MEAN_PATH = './deploy/mean_mnist.binaryproto'


def load_mean():
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(MEAN_PATH, 'rb').read()
    blob.ParseFromString(data)
    m = np.array(caffe.io.blobproto_to_array(blob))
    return m


def transform_mean(m):
    mean_img = np.zeros((3, 256, 256), np.float)
    for i in range(0, 256):
        for j in range(0, 256):
            mean_img[0][i][j] = m[0][0][i][j]
            mean_img[1][i][j] = m[0][1][i][j]
            mean_img[2][i][j] = m[0][2][i][j]
    return mean_img


def load_net(pretrained):
    net = caffe.Net(DEPLOY_FILE,
                    pretrained,
                    caffe.TEST)
    return net


def load_paths(filename):
    f = open(filename, 'r')
    lines = f.read().splitlines()
    lines = [(int(l.split()[1]), l.split()[0]) for l in lines]
    f.close()
    return lines


def print_final_results(ok, wrong, total, conf_matrix):
    print("\n"+"*"*100)
    print("Final Results\n")
    acc = ok / float(total)
    print("Total cases: {} - Cases OK: {}".format(total, ok))
    print("Accuracy: {0:.2f}".format(acc))
    print("Confussion Matrix:\n")
    for row in conf_matrix:
        total_class = sum(row)
        if total_class != 0:
            r = ["{0:.1f}".format(100*elem/float(total_class)) for elem in row]
        else:
            r = ["{0:.1f}".format(0) for elem in row]

        s = '\t'.join(r)
        r = ["{0}".format(elem) for elem in row]
        s += "\t\t\t"
        s += '\t'.join(r)
        print(s)
    print("Error rate: {}".format(wrong/float(total)))
    print("\n"+"*"*100)


def create_transformer(net, mean):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    #transformer.set_transpose('data', (0,))
    #transformer.set_mean('data', mean)
    # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_raw_scale('data', 255)
    # the reference model has channels in BGR order instead of RGB
    #transformer.set_channel_swap('data', (0,))
    return transformer


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("\nYou have to provide the snapshot to test. Example\n\
              \r{} snapshot_iter_6000.caffemodel\n".format(sys.argv[0]))
        sys.exit(1)

    pretrained = sys.argv[1]
    print(pretrained)
    paths = load_paths(IMGS_LIST)
    net = load_net(pretrained)
    transformer = create_transformer(net, None)

    ok = 0
    total = 0
    wrong = 0
    conf_matrix = [[0]*len(classes)for i in range(0, len(classes))]

    print("Starting testing...")
    start = time.time()
    for realclass, path in paths:
        try:
            img = caffe.io.load_image(os.path.join(ROOT_IMAGES, path))
            img = img[:,:,0]
            img = transformer.preprocess('data', img)
            net.blobs['data'].data[...] = img
            out = net.forward()
            classs = out['prob'][0].argmax()

            if classs == realclass:
                ok += 1
            else:
                wrong += 1

            conf_matrix[realclass][classs] += 1
            total += 1

            acc = ok / float(total)
            out = "\rTotal: {0}  - Accuracy: {1:.2f} - Error rate: {2:.3f}"
            sys.stdout.write(out.format(total, acc, wrong/float(total)))
            sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nTesting interrupted.\n")
            print_final_results(ok, wrong, total, conf_matrix)
            end = time.time()
            print("\nTotal elapsed time: {} seconds.".format(end-start))
            sys.exit(0)

    end = time.time()
    print_final_results(ok, wrong, total, conf_matrix)
    print("\nTotal elapsed time: {} seconds.".format(end-start))
    sys.exit(0)
