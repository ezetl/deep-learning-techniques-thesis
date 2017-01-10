#!/usr/bin/env python2.7
from os.path import join, exists
from optparse import OptionParser, OptionGroup
from utils.nets.cnn_factory import KITTINetFactory 
from utils.solver.solver import train_net, create_solver_params

"""
Script to reproduce the KITTI results + SUN dataset.
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
    parser.add_option("-L", "--lmdb-root", dest="lmdb_root", default='.',
            help="Root dir where all the LMDBs were created.", metavar="PATH")

    group_example = OptionGroup(parser, "Example:", 
            './experiment_kitti.py -L /media/eze/Datasets/KITTI/\\')
    parser.add_option_group(group_example)

    return parser.parse_args()


if __name__ == "__main__":
    (opts, args) = parse_options()

    acc = {'egomotion': {}, 'cont_10': {}, 'cont_100': {}}  

    scale = 1.0
    batch_size = 125
    # TODO: add pre-training with imagenet on AlexNet with 20K and 1M images

    ## EGOMOTION NET
    ## Used to train a siamese network from scratch following the method from the 
    ## paper
    siam_kitti, loss_blobs, acc_blobs = KITTINetFactory.siamese_egomotion(
            lmdb_path=join(opts.lmdb_root, 'KITTI/kitti_train_egomotion_lmdb'),
            labels_lmdb_path=join(opts.lmdb_root, 'KITTI/kitti_train_egomotion_lmdb_labels'),
            batch_size=batch_size,
            scale=scale,
            is_train=True,
            learn_all=True
            )

    # Create a SolverParameter instance with the predefined parameters for this experiment.
    # Some paths and iterations numbers will change for nets with contrastive loss or 
    # in finetuning stage 
    iters=60000
    ## Train our first siamese net with Egomotion method
    results_path = './results/kitti/'
    if exists(join(results_path, 'egomotion.npy')):
        results_ego = np.load(join(results_path, 'egomotion.npy'))
    else:
        results_ego = train_net(create_solver_params(siam_kitti, max_iter=iters, base_lr=0.001, snapshot_prefix='kitti/snapshots/egomotion/kitti_siamese'),
                loss_blobs=loss_blobs)
        np.save(join(results_path, 'egomotion.npy'))

    # CONTRASTIVE NET, m=10
    # Using a small batch size while training with Contrastive Loss leads 
    # to high bias in the networks (i.e. they dont learn much)
    # A good ad-hoc value is 500
    siam_cont10_kitti, loss_cont_blobs, acc_cont_blobs = KITTINetFactory.siamese_contrastive(
            lmdb_path=join(opts.lmdb_root, 'KITTI/kitti_train_egomotion_lmdb'),
            batch_size=batch_size,
            scale=scale,
            contrastive_margin=10,
            is_train=True,
            learn_all=True
            )
    # Also, using a big lr (i.e. 0.01) while training with Contrastive Loss can lead to nan values while backpropagating the loss
    if exists(join(results_path, 'contr_10.npy')):
        results_contr10 = np.load(join(results_path, 'contr_10.npy'))
    else:
        results_contr10 = train_net(create_solver_params(siam_cont10_kitti, max_iter=iters, base_lr=0.001, snapshot_prefix='kitti/snapshots/contrastive/kitti_siamese_m10'),
                loss_blobs=loss_cont_blobs)
        np.save(join(results_path, 'contr_10.npy'))

    ## CONTRASTIVE NET, m=100
    siam_cont100_kitti, loss_cont_blobs2, acc_cont_blobs2 = KITTINetFactory.siamese_contrastive(
            lmdb_path=join(opts.lmdb_root, 'KITTI/kitti_train_egomotion_lmdb'),
            batch_size=batch_size,
            scale=scale,
            contrastive_margin=100,
            is_train=True,
            learn_all=True
            )
    if exists(join(results_path, 'contr_100.npy')):
        results_contr100 = np.load(join(results_path, 'contr_100.npy'))
    else:
        results_contr100 = train_net(create_solver_params(siam_cont100_kitti, max_iter=iters,  base_lr=0.001, snapshot_prefix='kitti/snapshots/contrastive/kitti_siamese_m100'),
                loss_blobs=loss_cont_blobs2)
        np.save(join(results_path, 'contr_100.npy'))
    
    sizes_lmdb = ['5', '20'] 
    splits = ['01', '02', '03']
    outputs_to_test = ['1', '2', '3', '4', '5']
    iters = 10000 
    for output in outputs_to_test:
        acc['egomotion'][output] = {}
        acc['cont_10'][output] = {}
        acc['cont_100'][output] = {} 
        for num in sizes_lmdb:
            acc['egomotion'][output][num] = acc['cont_10'][output][num] = acc['cont_100'][output][num] = 0
            for split in splits:
                # Finetune network
                kitti_finetune, loss_blobs_f, acc_blobs_f = KITTINetFactory.standar(
                        lmdb_path=join(opts.lmdb_root, 'SUN397/lmdbs/SUN_Training_{}_{}perclass_lmdb'.format(split, num)),
                        batch_size=batch_size,
                        scale=scale,
                        num_classes=397,
                        is_train=True,
                        learn_all=False,
                        layers=output
                        )

                # Test Net Used to test accuracy in finetunig stages
                kitti_test, loss_blobs_test, acc_blobs_test = KITTINetFactory.standar(
                        lmdb_path=join(opts.lmdb_root, 'SUN397/lmdbs/SUN_Testing_{}_{}perclass_lmdb'.format(split, num)),
                        batch_size=batch_size,
                        scale=scale,
                        num_classes=397,
                        is_train=False,
                        learn_all=False,
                        layers=output
                        )

                # EGOMOTION
                snapshot_prefix = 'kitti/snapshots/egomotion_finetuning/kitti_lmdb{}'.format(num)
                results_egomotion = train_net(create_solver_params(kitti_finetune, test_netspec=kitti_test, max_iter=iters, test_interv=iters,
                                                                   base_lr=0.001, snapshot_prefix=snapshot_prefix),
                                              loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_f, pretrained_weights=results_ego['snaps'][-1])
                acc['egomotion'][output][num] += results_egomotion['acc'][acc_blobs_test[0]][0]

                # CONTRASTIVE m=10
                snapshot_prefix = 'kitti/snapshots/contrastive_finetuning10/kitti_lmdb{}'.format(num)
                results_contrastive10 = train_net(create_solver_params(kitti_finetune, test_netspec=kitti_test, max_iter=iters, test_interv=iters, base_lr=0.001, snapshot_prefix=snapshot_prefix), 
                                                  loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_test, pretrained_weights=results_contr10['snaps'][-1])
                acc['cont_10'][output][num] += results_contrastive10['acc'][acc_blobs_test[0]][0]

                # Contrastive m=100
                snapshot_prefix = 'kitti/snapshots/contrastive_finetuning100/kitti_lmdb{}'.format(num)
                results_contrastive100 = train_net(create_solver_params(kitti_finetune, test_netspec=kitti_test, max_iter=iters, base_lr=0.001, snapshot_prefix=snapshot_prefix), 
                                                   loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_test, pretrained_weights=results_contr100['snaps'][-1])
                acc['cont_100'][output][num] += results_contrastive100['acc'][acc_blobs_test[0]][0]


            acc['cont_100'][output][num] = acc['cont_100'][output][num] / float(len(splits)) 
            acc['cont_10'][output][num] = acc['cont_10'][output][num] / float(len(splits))
            acc['egomotion'][output][num] = acc['egomotion'][output][num] / float(len(splits)) 

    print('Accuracies')
    for a in ['cont_10', 'cont_100', 'egomotion']:
        res = a
        for num in sizes_lmdb:
            res += "   \t {}\t".format(num)
            for out in outputs_to_test:
                res += "    \t" + "{0:.3f}".format(acc[a][out][num])
        print(res)