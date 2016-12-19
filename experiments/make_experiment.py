#!/usr/bin/env python2.7
from optparse import OptionParser, OptionGroup
from utils.nets.cnn_factory import MNISTNetFactory, KITTINetFactory 
from utils.solver.solver import train_net 


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
    group_example = OptionGroup(parser, "Example:",
            ' '.join(['./make_experiment.py -b 125 -r -n 397\\',
                '-T /media/eze/Datasets/MNIST/mnist_train_siamese_lmdb\\',
                '-L /media/eze/Datasets/MNIST/mnist_train_siamese_lmdb_labels\\',
                '-s $(python -c"print(1/255.0)") -a']))

    parser.add_option_group(group)
    parser.add_option_group(group_example)

    (options, args) = parser.parse_args()

    alex, loss_blobs, acc_blobs = KITTINetFactory.standar(
            lmdb_path=options.lmdb_path,
            labels_lmdb_path=options.labels_lmdb_path,
            batch_size=options.batch_size,
            scale=options.scale,
            is_train=options.train,
            num_classes=options.num_classes,
            learn_all=options.train_all
            )

    siam_alex, loss_blobs, acc_blobs = KITTINetFactory.siamese_egomotion(
            lmdb_path=options.lmdb_path,
            labels_lmdb_path=options.labels_lmdb_path,
            batch_size=options.batch_size,
            scale=options.scale,
            is_train=options.train,
            learn_all=options.train_all
            )

    siam_mnist, loss_blobs, acc_blobs = MNISTNetFactory.siamese_egomotion(
            lmdb_path=options.lmdb_path,
            labels_lmdb_path=options.labels_lmdb_path,
            batch_size=options.batch_size,
            scale=options.scale,
            is_train=options.train,
            learn_all=options.train_all
            )

    mnist, loss_blobs_f, acc_blobs_f = MNISTNetFactory.standar(
            lmdb_path='/media/eze/Datasets/MNIST/mnist_standar_lmdb_10000',
            batch_size=options.batch_size,
            scale=options.scale,
            is_train=options.train,
            learn_all=False
            )

    mnist_test, loss_blobs_test, acc_blobs_test = MNISTNetFactory.standar(
            lmdb_path='/media/eze/Datasets/MNIST/mnist_standar_lmdb_test',
            batch_size=options.batch_size,
            scale=options.scale,
            is_train=False,
            learn_all=False
            )

    #niter = 40000
    #print 'Running solver for {} iterations...'.format(niter)
    #results = train_net(siam_mnist, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs, snapshot_prefix='mnist/snapshots/egomotion/mnist_siamese')
    #print(results['snaps'])
    #niter = 8000
    #acc = 0
    #results_f = train_net(mnist, test_netspec=mnist_test, test_interv=niter, test_iter=80, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_f, pretrained_weights=results['snaps'][-1], snapshot_prefix='mnist/snapshots/finetuning/mnist')
    #acc += results_f['acc'][acc_blobs_test[0]][0]
    #results_f = train_net(mnist, test_netspec=mnist_test, test_interv=niter, test_iter=80, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_f, pretrained_weights=results['snaps'][-1], snapshot_prefix='mnist/snapshots/finetuning/mnist')
    #acc += results_f['acc'][acc_blobs_test[0]][0]
    #results_f = train_net(mnist, test_netspec=mnist_test, test_interv=niter, test_iter=80, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_f, pretrained_weights=results['snaps'][-1], snapshot_prefix='mnist/snapshots/finetuning/mnist')
    #acc += results_f['acc'][acc_blobs_test[0]][0]
    #print("Final accuracy Siamese+finetuning (avg. of 3 runs): {0:.2f}".format(acc/3.0))

    niter=10000
    mnist, loss_blobs_f, acc_blobs_f = MNISTNetFactory.standar(
            lmdb_path='/media/eze/Datasets/MNIST/mnist_standar_lmdb_1000',
            batch_size=options.batch_size,
            scale=options.scale,
            is_train=True,
            learn_all=True
            )
    mnist_test, loss_blobs_test, acc_blobs_test = MNISTNetFactory.standar(
            lmdb_path='/media/eze/Datasets/MNIST/mnist_standar_lmdb_test',
            batch_size=options.batch_size,
            scale=options.scale,
            is_train=False,
            learn_all=False
            )
    acc = 0
    results_f = train_net(mnist, test_netspec=mnist_test, test_interv=niter, test_iter=80, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_test, snapshot_prefix='mnist/snapshots/standar/mnist')
    acc += results_f['acc'][acc_blobs_test[0]][0]
    results_f = train_net(mnist, test_netspec=mnist_test, test_interv=niter, test_iter=80, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_test, snapshot_prefix='mnist/snapshots/standar/mnist')
    acc += results_f['acc'][acc_blobs_test[0]][0]
    results_f = train_net(mnist, test_netspec=mnist_test, test_interv=niter, test_iter=80, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_test, snapshot_prefix='mnist/snapshots/standar/mnist')
    acc += results_f['acc'][acc_blobs_test[0]][0]
    print("Final error in standar net (avg. of 3 runs): {0:.3f}. Acc: {1:.3f}".format(1 - acc/3.0, acc/3.0))
