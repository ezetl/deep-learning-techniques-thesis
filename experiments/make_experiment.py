#!/usr/bin/env python2.7
from optparse import OptionParser, OptionGroup
from utils.cnn_factory import *
from utils.solver_factory import *


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
    group.add_option("-T", "--train-lmdb", dest="lmdb_path",
            help="LMDB with train data", metavar="PATH")
    group.add_option("-L", "--train-labels-lmdb", dest="labels_lmdb_path",
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
            lmdb_path=options.lmdb_path,
            labels_lmdb_path=options.labels_lmdb_path,
            batch_size=options.batch_size,
            scale=options.scale,
            train=options.train,
            num_classes=options.num_classes,
            learn_all=options.train_all
            )

    siam_alex = siamese_alexnet_kitti(
            lmdb_path=options.lmdb_path,
            labels_lmdb_path=options.labels_lmdb_path,
            batch_size=options.batch_size,
            scale=options.scale,
            train=options.train,
            learn_all=options.train_all
            )

    siam_mnist = siamese_alexnet_mnist(
            lmdb_path=options.lmdb_path,
            labels_lmdb_path=options.labels_lmdb_path,
            batch_size=options.batch_size,
            scale=options.scale,
            train=options.train,
            learn_all=options.train_all
            )

    # write the nets
    alex_file = 'alexnet.prototxt'
    with open(alex_file, 'w') as f:
        f.write(str(alex.to_proto()))

    siam_kitti = 'siamese_alexnet.prototxt'
    with open(siam_kitti, 'w') as f:
        f.write(str(siam_alex.to_proto()))

    siammnist_file = 'siamese_mnist.prototxt'
    with open(siammnist_file, 'w') as f:
        f.write(str(siam_mnist.to_proto()))

    niter = 40000
    print 'Running solver for {} iterations...'.format(niter)
    #loss = run_solver(create_solver_settings(siammnist_file, max_iter=niter, stepsize=10000,  snapshot_prefix='mnist/snapshots/egomotion/mnist_siamese'), max_iters=niter)
    loss = run_solver(create_solver_settings(siam_kitti, max_iter=niter, stepsize=10000,  snapshot_prefix='mnist/snapshots/egomotion/mnist_siamese'), max_iters=niter)
    print(loss)
