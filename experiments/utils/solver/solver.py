from os import makedirs
from os.path import join, exists, dirname
from collections import defaultdict
import pickle
from sys import exit
import tempfile
import numpy as np
import caffe
from caffe.proto import caffe_pb2

MIN_LOSS = 1000000

class TrainException(Exception):
    pass

def create_solver_params(train_netspec, test_netspec=None, test_interv=1000, test_iter=80, base_lr=0.01,
    max_iter=40000, lr_policy='step', stepsize=10000, gamma=0.5, snapshot=1000, snapshot_prefix='/tmp/snapshot'):
    """
    Code to create a solver. Taken and modified from
    this Caffe tutorial: https://github.com/BVLC/caffe/blob/master/examples/02-fine-tuning.ipynb
    :param train_net_path: str. Path to the train protoxt
    :param test_net_path: str. Path to the test protoxt
    :param test_interv: int. Test will be performed ever test_interv iterations. Only useful if test_net_path is passed.
    :param base_lr: float. Initial learning rate
    :param max_iter: int. Maximum amount of iterations for training
    :param stepsize:  int. Amount of iterations between each lr update.
    :returns: str. The name of the tmp file where the Solver was stored 
    """
    solver = caffe_pb2.SolverParameter()

    """
    TODO: try this
    test_compute_loss: true
    test_initialization: true
    """
    # Specify locations of the train and (maybe) test networks.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(train_netspec.to_proto()))
        solver.train_net = f.name

    if test_netspec:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(str(test_netspec.to_proto()))
            solver.test_net.append(f.name)
            solver.test_interval = min(test_interv, max_iter) 
            solver.test_iter.append(test_iter)

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    solver.iter_size = 1
    
    solver.max_iter = max_iter
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    solver.type = 'SGD'
    
    # Set the initial learning rate for SGD.
    solver.base_lr = base_lr
    
    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    solver.lr_policy = lr_policy 
    if lr_policy == 'step':
        solver.gamma = gamma 
        solver.stepsize = stepsize
    
    solver.snapshot = min(snapshot, max_iter)
    solver.snapshot_prefix = snapshot_prefix
    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    solver.momentum = 0.9
    solver.weight_decay = 5e-4
    
    # Display the current training loss and accuracy every 100 iterations.
    solver.display = 100
    
    # Train on the GPU.
    solver.solver_mode = caffe_pb2.SolverParameter.GPU
    
    return solver


def train_net(solver_param, loss_blobs=None, acc_blobs=None, pretrained_weights="", pickle_name="./pickle.pickle"):
    """
    Run a solver instance from solver_params for solver_params.max_iter iterations,
    returning the loss and accuracy recorded in each step.

    :param solver_params: Caffe SolverParameter instance
    :param loss_blobs: tuple of str. The names of the loss layers used to gather the loss info during training. These losses will then be returned as a result
    :param acc_blobs: tuple of str. Same as loss_blobs but related to accuracies 
    :param pretrained_weights: str. Path where the pretrained .caffemodel is. If not set, train from scratch is performed.
    :param pickle_name: str. Path were the results (loss, paths to models) are going to be stored as a pickle file.
                        If the pickle file already exists, then it loads it in memory and returns the precomputed results

    :return: A dict containing at most 3 keys:
              {
                'snaps': ['path/to/snapshot1', 'path/to/snapshot2', ... ],
                'loss': {
                           'loss_x': list containing the training 'loss_x' captured in every iteration,
                           'loss_y': list ...
                        },
                 'acc': {
                           'acc_x': list containing some accuracy captured every 'test_inverv' iterations,
                           ...
                        } 
              }
    """
    if not exists(dirname(solver_param.snapshot_prefix)):
        print("Path for snapshots does not exist. Creating dir {}".format(dirname(solver_param.snapshot_prefix)))
        makedirs(dirname(solver_param.snapshot_prefix))

    #with tempfile.NamedTemporaryFile(delete=False) as f:
    with open('/home/eze/Projects/tesis/experiments/solver.prototxt', 'w') as f:
        f.write(str(solver_param))
        solver_param_fname = f.name

    solver = caffe.get_solver(solver_param_fname)

    if pretrained_weights and not exists(pretrained_weights):
        raise TrainException("Could not find pretrained weights with provided path: {}".format(pretrained_weights)) 
    elif exists(pretrained_weights):
        print("Loading weights from {}".format(pretrained_weights))
        solver.net.copy_from(pretrained_weights)

    results = {'snaps': [], 'best_snap': None} 

    if loss_blobs:
        results['loss'] = {loss_name: np.array([]) for loss_name in loss_blobs}
    else:
        results['loss'] = {'loss': np.array([])}

    if acc_blobs is not None:
        results['acc'] = {} 
        results['best_acc'] = defaultdict(np.float32) 
    else:
        acc_blobs = []

    init_iter = 0

    if exists(pickle_name):
        with open(pickle_name, 'rb') as handle:
            results = pickle.load(handle)
            init_iter = int(results['snaps'][-1].split('iter_')[1].replace('.caffemodel','')) 
            if solver_param.max_iter <= init_iter: 
                print("This model has already been trained. Returning saved pickle")
                # This training session has already finished, return the results
                return results
            else:
                solverstate = results['snaps'][-1].replace('caffemodel', 'solverstate')
                if exists(solverstate):
                    print("Keep training model from iteration {}".format(init_iter))
                    solver.restore(solverstate)
                else:
                    print("The .solverstate for iteration {} could not be found. Training again.".format(init_iter))
    
    min_loss = MIN_LOSS
    try:
        for it in range(init_iter, solver_param.max_iter+1):
            solver.step(1)
            # Retrieve loss of this step
            total_loss = 0
            for name in loss_blobs:
                partial_loss = solver.net.blobs[name].data.item()
                results['loss'][name] = np.append(results['loss'].get(name, np.array([])), partial_loss)
                # if we have several losses (like translations, rotations, etc) we must aggregate them
                total_loss += partial_loss

            # Retrieve accuracy of tests every 'test_interval' iterations
            if solver_param.test_interval and it!=0 and it % solver_param.test_interval == 0:
                for name in acc_blobs:
                    acc = solver.test_nets[0].blobs[name].data.item()
                    results['acc'][name] = np.append(results['acc'].get(name, np.array([])), acc)
                    if results['best_acc'][name] <= acc:
                        results['best_acc'][name] = acc

            # Save snapshots names
            if it!=0 and it % solver_param.snapshot == 0:
                snapshot_name = str(solver_param.snapshot_prefix + '_iter_{}.caffemodel'.format(it))
                if min_loss > total_loss:
                    min_loss = total_loss
                    results['best_snap'] = snapshot_name
                print("Best snapshot so far: Iteration {}, {}".format(it, results['best_snap']))    
                results['snaps'].append(snapshot_name) 
                with open(pickle_name, 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except KeyboardInterrupt:
        exit("Training has been interrupted. Bye!")

    with open(pickle_name, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results 
