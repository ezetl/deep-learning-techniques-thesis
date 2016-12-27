#!/usr/bin/env python2.7
from os.path import join
from optparse import OptionParser, OptionGroup
from utils.nets.cnn_factory import MNISTNetFactory, KITTINetFactory 
from utils.solver.solver import train_net 

"""
Script to reproduce the MNIST results.

I thought about creating some classes to 'wrap' the experiments, some
sort of Experiment class which would contain and correctly instantiate
all the networks, LMDBs' paths and hyperparameters. Then I realized
that the large amount of parameters that have to be set for each
training can become a mess to manage.

I concluded that it is better and faster to just use some auxiliary
functions to create, train and finetune the networks and then create 
all the needed scripts using those 'building blocks'.

By doing that, setting the parameters for each 'subexperiment'
(training, finetuning) is more explicit. Also, as this is just an
intent to reproduce a paper, we are not intending to explore all the
possible hyperparameters and architectures of a CNN, and so I think 
this is a reasonable trade off between generality and readability.
"""


def parse_options():
    """
    Parse the command line options and returns the (options, arguments)
    tuple.

    The option -L is because I assume you have created all the LMDBs in 
    the same folder. That makes things easier. Also, I assume you didn't 
    change the original LMDB names (as they appear in the preprocessing
    scripts). If you did different, then feel free to change the 
    lmdb_path and labels_lmdb_path parameters in the MNISTNetFactory 
    and KITTINetFactory's methods calls
    """
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-b", "--batch-size", dest="batch_size", type="int",
            default=125, help="Batch size", metavar="INT")
    parser.add_option("-i", "--train-iters", dest="train_iters", type="int",
            default=40000, help="Iterations for training", metavar="INT")
    parser.add_option("-f", "--finetune-iters", dest="finetune_iters", type="int",
            default=4000, help="Iterations for finetuning", metavar="INT")
    parser.add_option("-s", "--scale", dest="scale", type="float", default=1.0,
            help="Scale param. For example, if you want to provide a way to transform your images from [0,255] to [0,1] range", metavar="FLOAT")
    parser.add_option("-L", "--lmdb-root", dest="lmdb_root",
            help="Root dir where all the LMDBs were created.", metavar="PATH")

    group_example = OptionGroup(parser, "Example:", 
            './make_experiment.py -b 125 -L /media/eze/Datasets/MNIST/ -s $(python -c"print(1/255.0)") -i 40000 -f 4000\\')
    parser.add_option_group(group_example)

    return parser.parse_args()


if __name__ == "__main__":
    (opts, args) = parse_options()

    acc = {'ego': {}, 'cont_10': {}, 'cont_100': {}, 'stand': {}}  

    # TEST NET
    mnist_test, loss_blobs_test, acc_blobs_test = MNISTNetFactory.standar(
            lmdb_path=join(opts.lmdb_root, 'mnist_standar_lmdb_test'),
            batch_size=opts.batch_size,
            scale=opts.scale,
            is_train=False,
            learn_all=False
            )

    # EGOMOTION NET
    siam_mnist, loss_blobs, acc_blobs = MNISTNetFactory.siamese_egomotion(
            lmdb_path=join(opts.lmdb_root, 'mnist_train_siamese_lmdb'),
            labels_lmdb_path=join(opts.lmdb_root, 'mnist_train_siamese_lmdb_labels'),
            batch_size=opts.batch_size,
            scale=opts.scale,
            is_train=True,
            learn_all=True
            )
    results_ego = train_net(siam_mnist, max_iter=opts.train_iters, stepsize=10000, loss_blobs=loss_blobs, snapshot_prefix='mnist/snapshots/egomotion/mnist_siamese')

    # CONTRASTIVE NET, m=10
    # Using a small batch size while training with Contrastive Loss leads 
    # to high bias in the networks (i.e. they dont learn much)
    # A good ad-hoc value is 500
    siam_cont_mnist, loss_cont_blobs, acc_cont_blobs = MNISTNetFactory.siamese_contrastive(
            lmdb_path=join(opts.lmdb_root, 'mnist_train_siamese_lmdb'),
            batch_size=500,
            scale=opts.scale,
            contrastive_margin=10,
            is_train=True,
            learn_all=True
            )
    # Also, using a big lr (i.e. 0.01) whule training with Contrastive Loss can lead to nan values while backpropagating the loss
    results_contr10 = train_net(siam_cont_mnist, max_iter=opts.train_iters, stepsize=10000, base_lr=0.001, loss_blobs=loss_cont_blobs, snapshot_prefix='mnist/snapshots/contrastive/mnist_siamesem10')

    # CONTRASTIVE NET, m=100
    siam_cont_mnist2, loss_cont_blobs2, acc_cont_blobs2 = MNISTNetFactory.siamese_contrastive(
            lmdb_path=join(opts.lmdb_root, 'mnist_train_siamese_lmdb'),
            batch_size=500,
            scale=opts.scale,
            contrastive_margin=100,
            is_train=True,
            learn_all=True
            )
    results_contr100 = train_net(siam_cont_mnist2, max_iter=opts.train_iters, stepsize=10000, base_lr=0.001, loss_blobs=loss_cont_blobs2, snapshot_prefix='mnist/snapshots/contrastive/mnist_siamesem100')
    
    repeat = 3
    sizes_lmdb = ['100', '300', '1000', '10000'] 
    for num in sizes_lmdb:
        acc['ego'][num] = 0
        acc['stand'][num] = 0
        acc['cont_10'][num] = 0
        acc['cont_100'][num] = 0

        # Finetune network
        mnist_finetune, loss_blobs_f, acc_blobs_f = MNISTNetFactory.standar(
                lmdb_path=join(opts.lmdb_root, 'mnist_standar_lmdb_{}'.format(num)),
                batch_size=opts.batch_size,
                scale=opts.scale,
                is_train=True,
                learn_all=False
                )

        # Train from scratch network
        mnist, loss_blobs_st, acc_blobs_st = MNISTNetFactory.standar(
                lmdb_path=join(opts.lmdb_root, 'mnist_standar_lmdb_{}'.format(num)),
                batch_size=opts.batch_size,
                scale=opts.scale,
                is_train=True,
                learn_all=True
                )

        for i in range(0, repeat):
            # EGOMOTION
            results_egomotion = train_net(mnist_finetune, test_netspec=mnist_test, test_interv=opts.finetune_iters, test_iter=80, max_iter=opts.finetune_iters, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_f, pretrained_weights=results_ego['snaps'][-1], snapshot_prefix='mnist/snapshots/finetuning/mnist')
            acc['ego'][num] += results_egomotion['acc'][acc_blobs_test[0]][0]

            # STANDAR
            results_standar = train_net(mnist, test_netspec=mnist_test, test_interv=opts.train_iters, test_iter=80, max_iter=opts.train_iters, stepsize=10000, loss_blobs=loss_blobs_st, acc_blobs=acc_blobs_st, snapshot_prefix='mnist/snapshots/standar/mnist')
            acc['stand'][num] += results_standar['acc'][acc_blobs_test[0]][0]

            # CONTRASTIVE m=10
            results_contrastive10 = train_net(mnist_finetune, test_netspec=mnist_test, test_interv=opts.finetune_iters, test_iter=80, max_iter=opts.finetune_iters, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_test,
                    pretrained_weights=results_contr10['snaps'][-1], snapshot_prefix='mnist/snapshots/contrastive_finetune10/mnist')
            acc['cont_10'][num] += results_contrastive10['acc'][acc_blobs_test[0]][0]

            # Contrastive m=100
            results_contrastive100 = train_net(mnist_finetune, test_netspec=mnist_test, test_interv=opts.finetune_iters, test_iter=80, max_iter=opts.finetune_iters, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_test,
                pretrained_weights=results_contr100['snaps'][-1], snapshot_prefix='mnist/snapshots/contrastive_finetune100/mnist')
            acc['cont_100'][num] += results_contrastive100['acc'][acc_blobs_test[0]][0]

        acc['cont_100'][num] = acc['cont_100'][num] / float(repeat) 
        acc['cont_10'][num] = acc['cont_10'][num] / float(repeat)
        acc['ego'][num] = acc['ego'][num] / float(repeat) 
        acc['stand'][num] = acc['stand'][num] / float(repeat)

    print('Accuracies')
    for a in ['stand', 'cont_10', 'cont_100', 'ego']:
        print('{}\t{}'.format(a, '\t'.join([str(acc[a]['100']), str(acc[a]['300']), str(acc[a]['1000']), str(acc[a]['10000'])])))
