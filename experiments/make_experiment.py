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

    #alex, loss_blobs, acc_blobs = KITTINetFactory.standar(
    #        lmdb_path=options.lmdb_path,
    #        labels_lmdb_path=options.labels_lmdb_path,
    #        batch_size=options.batch_size,
    #        scale=options.scale,
    #        is_train=options.train,
    #        num_classes=options.num_classes,
    #        learn_all=options.train_all
    #        )

    #siam_alex, loss_blobs, acc_blobs = KITTINetFactory.siamese_egomotion(
    #        lmdb_path=options.lmdb_path,
    #        labels_lmdb_path=options.labels_lmdb_path,
    #        batch_size=options.batch_size,
    #        scale=options.scale,
    #        is_train=options.train,
    #        learn_all=options.train_all
    #        )

    acc = {'ego': {}, 'cont_10': {}, 'cont_100': {}, 'stand': {}}  

    ### TEST
    mnist_test, loss_blobs_test, acc_blobs_test = MNISTNetFactory.standar(
            lmdb_path='/media/eze/Datasets/MNIST/mnist_standar_lmdb_test',
            batch_size=options.batch_size,
            scale=options.scale,
            is_train=False,
            learn_all=False
            )

    ### EGOMOTION
    niter = 40000
    siam_mnist, loss_blobs, acc_blobs = MNISTNetFactory.siamese_egomotion(
            lmdb_path=options.lmdb_path,
            labels_lmdb_path=options.labels_lmdb_path,
            batch_size=options.batch_size,
            scale=options.scale,
            is_train=options.train,
            learn_all=options.train_all
            )
    results_ego = train_net(siam_mnist, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs, snapshot_prefix='mnist/snapshots/egomotion/mnist_siamese')

    ### CONTRASTIVE, m=10
    siam_cont_mnist, loss_cont_blobs, acc_cont_blobs = MNISTNetFactory.siamese_contrastive(
            lmdb_path=options.lmdb_path,
            batch_size=options.batch_size,
            scale=options.scale,
            contrastive_margin=10,
            is_train=options.train,
            learn_all=options.train_all
            )
    results_contr10 = train_net(siam_cont_mnist, max_iter=niter, stepsize=10000, base_lr=0.001, loss_blobs=loss_cont_blobs, snapshot_prefix='mnist/snapshots/contrastive/mnist_siamesem10')

    ### CONTRASTIVE m=100
    siam_cont_mnist2, loss_cont_blobs2, acc_cont_blobs2 = MNISTNetFactory.siamese_contrastive(
            lmdb_path=options.lmdb_path,
            batch_size=500,
            scale=options.scale,
            contrastive_margin=100,
            is_train=options.train,
            learn_all=options.train_all
            )
    results_contr100 = train_net(siam_cont_mnist2, max_iter=niter, stepsize=10000, base_lr=0.001, loss_blobs=loss_cont_blobs2, snapshot_prefix='mnist/snapshots/contrastive/mnist_siamesem100')

    
    niter = 5000
    sniter = 40000
    sizes_lmdb = ['100', '300', '1000', '10000'] 
    for num in sizes_lmdb:
        acc['ego'][num] = 0
        acc['stand'][num] = 0
        acc['cont_10'][num] = 0
        acc['cont_100'][num] = 0

        mnist, loss_blobs_f, acc_blobs_f = MNISTNetFactory.standar(
                lmdb_path='/media/eze/Datasets/MNIST/mnist_standar_lmdb_{}'.format(num),
                batch_size=options.batch_size,
                scale=options.scale,
                is_train=True,
                learn_all=False
                )

        # EGOMOTION
        results_egomotion = train_net(mnist, test_netspec=mnist_test, test_interv=niter, test_iter=80, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_f, pretrained_weights=results_ego['snaps'][-1], snapshot_prefix='mnist/snapshots/finetuning/mnist')
        acc['ego'][num] += results_egomotion['acc'][acc_blobs_test[0]][0]
        results_egomotion = train_net(mnist, test_netspec=mnist_test, test_interv=niter, test_iter=80, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_f, pretrained_weights=results_ego['snaps'][-1], snapshot_prefix='mnist/snapshots/finetuning/mnist')
        acc['ego'][num] += results_egomotion['acc'][acc_blobs_test[0]][0]
        results_egomotion = train_net(mnist, test_netspec=mnist_test, test_interv=niter, test_iter=80, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_f, pretrained_weights=results_ego['snaps'][-1], snapshot_prefix='mnist/snapshots/finetuning/mnist')
        acc['ego'][num] += results_egomotion['acc'][acc_blobs_test[0]][0]
        acc['ego'][num] = acc['ego'][num] / 3.0

        ### STANDAR. Train from scratch for every dataset
        mnist, loss_blobs_st, acc_blobs_st = MNISTNetFactory.standar(
                lmdb_path='/media/eze/Datasets/MNIST/mnist_standar_lmdb_{}'.format(num),
                batch_size=options.batch_size,
                scale=options.scale,
                is_train=True,
                learn_all=True
                )
        results_standar = train_net(mnist, test_netspec=mnist_test, test_interv=sniter, test_iter=80, max_iter=sniter, stepsize=10000, loss_blobs=loss_blobs_st, acc_blobs=acc_blobs_st, snapshot_prefix='mnist/snapshots/standar/mnist')
        acc['stand'][num] += results_standar['acc'][acc_blobs_test[0]][0]
        results_standar = train_net(mnist, test_netspec=mnist_test, test_interv=sniter, test_iter=80, max_iter=sniter, stepsize=10000, loss_blobs=loss_blobs_st, acc_blobs=acc_blobs_st, snapshot_prefix='mnist/snapshots/standar/mnist')
        acc['stand'][num] += results_standar['acc'][acc_blobs_test[0]][0]
        results_standar = train_net(mnist, test_netspec=mnist_test, test_interv=sniter, test_iter=80, max_iter=sniter, stepsize=10000, loss_blobs=loss_blobs_st, acc_blobs=acc_blobs_st, snapshot_prefix='mnist/snapshots/standar/mnist')
        acc['stand'][num] += results_standar['acc'][acc_blobs_test[0]][0]
        acc['stand'][num] = acc['stand'][num] / 3.0

        # CONTRASTIVE m=10
        results_contrastive10 = train_net(mnist, test_netspec=mnist_test, test_interv=niter, test_iter=80, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_test,
                pretrained_weights=results_contr10['snaps'][-1], snapshot_prefix='mnist/snapshots/contrastive_finetune10/mnist')
        acc['cont_10'][num] += results_contrastive10['acc'][acc_blobs_test[0]][0]
        results_contrastive10 = train_net(mnist, test_netspec=mnist_test, test_interv=niter, test_iter=80, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_test,
                pretrained_weights=results_contr10['snaps'][-1], snapshot_prefix='mnist/snapshots/contrastive_finteune10/mnist')
        acc['cont_10'][num] += results_contrastive10['acc'][acc_blobs_test[0]][0]
        results_contrastive10 = train_net(mnist, test_netspec=mnist_test, test_interv=niter, test_iter=80, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_test,
                pretrained_weights=results_contr10['snaps'][-1], snapshot_prefix='mnist/snapshots/contrastive_finetune10/mnist')
        acc['cont_10'][num] += results_contrastive10['acc'][acc_blobs_test[0]][0]
        acc['cont_10'][num] = acc['cont_10'][num] / 3.0

        # Contrastive m=100
        results_contrastive100 = train_net(mnist, test_netspec=mnist_test, test_interv=niter, test_iter=80, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_test,
                pretrained_weights=results_contr100['snaps'][-1], snapshot_prefix='mnist/snapshots/contrastive_finetune100/mnist')
        acc['cont_100'][num] += results_contrastive100['acc'][acc_blobs_test[0]][0]
        results_contrastive100 = train_net(mnist, test_netspec=mnist_test, test_interv=niter, test_iter=80, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_test,
                pretrained_weights=results_contr100['snaps'][-1], snapshot_prefix='mnist/snapshots/contrastive_finteune100/mnist')
        acc['cont_100'][num] += results_contrastive100['acc'][acc_blobs_test[0]][0]
        results_contrastive100 = train_net(mnist, test_netspec=mnist_test, test_interv=niter, test_iter=80, max_iter=niter, stepsize=10000, loss_blobs=loss_blobs_f, acc_blobs=acc_blobs_test,
                pretrained_weights=results_contr100['snaps'][-1], snapshot_prefix='mnist/snapshots/contrastive_finetune100/mnist')
        acc['cont_100'][num] += results_contrastive100['acc'][acc_blobs_test[0]][0]
        acc['cont_100'][num] = acc['cont_100'][num] / 3.0

    print('Accuracies')
    for a in ['stand', 'cont_10', 'cont_100', 'ego']:
        print('{}\t{}'.format(a, '\t'.join([str(acc[a]['100']), str(acc[a]['300']), str(acc[a]['1000']), str(acc[a]['10000'])])))

