#!/usr/bin/env python2.7
from os.path import join, exists
from os import makedirs
import pickle
from optparse import OptionParser, OptionGroup
from utils.nets.cnn_factory import MNISTNetFactory, KITTINetFactory 
from utils.solver.solver import train_net, create_solver_params

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
    parser.add_option("-L", "--lmdb-root", dest="lmdb_root",
            help="Root dir where all the LMDBs were created.", metavar="PATH")

    group_example = OptionGroup(parser, "Example:", 
            './experiment_mnist.py -L /media/eze/Datasets/MNIST/\\')
    parser.add_option_group(group_example)

    return parser.parse_args()


if __name__ == "__main__":
    (opts, args) = parse_options()

    acc = {'ego': {}, 'cont_10': {}, 'cont_100': {}, 'stand': {}}  

    scale = 1/255.0
    results_path = './results/mnist/'
    try:
        makedirs(results_path)
    except:
        pass
    # TEST NET
    # Used to test accuracy in finetunig stages
    mnist_test, loss_blobs_test, acc_blobs_test = MNISTNetFactory.standar(
            lmdb_path=join(opts.lmdb_root, 'mnist_standar_lmdb_test'),
            batch_size=125,
            scale=scale,
            is_train=False,
            learn_all=False
            )

    ## EGOMOTION NET
    # Used to train a siamese network from scratch following the method from the 
    # paper
    siam_mnist, loss_blobs, acc_blobs = MNISTNetFactory.siamese_egomotion(
            lmdb_path=join(opts.lmdb_root, 'mnist_train_siamese_lmdb'),
            labels_lmdb_path=join(opts.lmdb_root, 'mnist_train_siamese_lmdb_labels'),
            batch_size=125,
            scale=scale,
            is_train=True,
            learn_all=True
            )

    # Create a SolverParameter instance with the predefined parameters for this experiment.
    # Some paths and iterations numbers will change for nets with contrastive loss or 
    # in finetuning stage 
    iters=40000
    # Train our first siamese net with Egomotion method
    results_ego = train_net(create_solver_params(siam_mnist, max_iter=iters, snapshot_prefix='snapshots/mnist/egomotion/mnist_siamese'),
            loss_blobs=loss_blobs, pickle_name=join(results_path, 'egomotion.pickle'))
    del siam_mnist        

    # CONTRASTIVE NET, m=10
    # Using a small batch size while training with Contrastive Loss leads 
    # to high bias in the networks (i.e. they dont learn much)
    # A good ad-hoc value is 500
    siam_cont10_mnist, loss_cont_blobs, acc_cont_blobs = MNISTNetFactory.siamese_contrastive(
            lmdb_path=join(opts.lmdb_root, 'mnist_train_siamese_lmdb'),
            batch_size=800,
            scale=scale,
            contrastive_margin=10,
            is_train=True,
            learn_all=True
            )
    # Also, using a big lr (i.e. 0.01) while training with Contrastive Loss can lead to nan values while backpropagating the loss
    results_contr10 = train_net(create_solver_params(siam_cont10_mnist, max_iter=iters, base_lr=0.01, snapshot_prefix='snapshots/mnist/contrastive/mnist_siamese_m10'),
            loss_blobs=loss_cont_blobs, pickle_name=join(results_path, 'contr_10.pickle'))
    del siam_cont10_mnist        

    # CONTRASTIVE NET, m=100
    siam_cont100_mnist, loss_cont_blobs2, acc_cont_blobs2 = MNISTNetFactory.siamese_contrastive(
            lmdb_path=join(opts.lmdb_root, 'mnist_train_siamese_lmdb'),
            batch_size=800,
            scale=scale,
            contrastive_margin=100,
            is_train=True,
            learn_all=True
            )
    results_contr100 = train_net(create_solver_params(siam_cont100_mnist, max_iter=iters,  base_lr=0.001, snapshot_prefix='snapshots/mnist/contrastive/mnist_siamese_m100'),
            loss_blobs=loss_cont_blobs2, pickle_name=join(results_path, 'contr_100.pickle'))
    del siam_cont100_mnist        
    
    repeat = 3
    sizes_lmdb = ['100', '300', '1000', '10000'] 
    iters = 4000 
    for num in sizes_lmdb:
        for k in acc:
            acc[k][output] = defaultdict(int)

        # Finetune network
        mnist_finetune, loss_blobs_f, acc_blobs_f = MNISTNetFactory.standar(
                lmdb_path=join(opts.lmdb_root, 'mnist_standar_lmdb_{}'.format(num)),
                batch_size=125,
                scale=scale,
                is_train=True,
                learn_all=False
                )

        # Train from scratch network
        mnist, loss_blobs_st, acc_blobs_st = MNISTNetFactory.standar(
                lmdb_path=join(opts.lmdb_root, 'mnist_standar_lmdb_{}'.format(num)),
                batch_size=125,
                scale=scale,
                is_train=True,
                learn_all=True
                )

        for i in range(0, repeat):
            # EGOMOTION
            snapshot_prefix = 'snapshots/mnist/egomotion_finetuning/mnist_repeat{}_lmdb{}'.format(i, num)
            results_egomotion = train_net(create_solver_params(mnist_finetune, test_netspec=mnist_test, max_iter=iters, test_interv=iters,
                                                               base_lr=0.01, snapshot_prefix=snapshot_prefix),
                                          loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_f, pretrained_weights=results_ego['best_snap'])
            acc['ego'][num] += results_egomotion['acc'][acc_blobs_test[0]][0]

            # CONTRASTIVE m=10
            snapshot_prefix = 'snapshots/mnist/contrastive_finetuning10/mnist_repeat{}_lmdb{}'.format(i, num)
            results_contrastive10 = train_net(create_solver_params(mnist_finetune, test_netspec=mnist_test, max_iter=iters, test_interv=iters, base_lr=0.01, snapshot_prefix=snapshot_prefix), 
                                              loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_test, pretrained_weights=results_contr10['best_snap'])
            acc['cont_10'][num] += results_contrastive10['acc'][acc_blobs_test[0]][0]

            # Contrastive m=100
            snapshot_prefix = 'snapshots/mnist/contrastive_finetuning100/mnist_repeat{}_lmdb{}'.format(i, num)
            results_contrastive100 = train_net(create_solver_params(mnist_finetune, test_netspec=mnist_test, max_iter=iters, base_lr=0.01, snapshot_prefix=snapshot_prefix), 
                                               loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_test, pretrained_weights=results_contr100['best_snap'])
            acc['cont_100'][num] += results_contrastive100['acc'][acc_blobs_test[0]][0]

            # STANDAR
            snapshot_prefix = 'snapshots/mnist/standar/mnist_repeat{}_lmdb{}'.format(i, num)
            results_standar = train_net(create_solver_params(mnist, test_netspec=mnist_test, base_lr=0.01, max_iter=40000, test_interv=40000, snapshot_prefix=snapshot_prefix),
                                        loss_blobs=loss_blobs_st, acc_blobs=acc_blobs_st)
            acc['stand'][num] += results_standar['acc'][acc_blobs_test[0]][0]

        for k in acc:
            acc[k][output][num] = acc[k][num] / float(len(repeat))

    print('Accuracies')
    for k in acc:
        print('{}   \t{}'.format(k, '\t'.join([str(acc[k]['100']), str(acc[k]['300']), str(acc[k]['1000']), str(acc[k]['10000'])])))

