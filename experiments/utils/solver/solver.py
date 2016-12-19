from os import makedirs
from os.path import join, exists, dirname
from sys import exit
import tempfile
import numpy as np
import caffe
from caffe.proto import caffe_pb2

class TrainException(Exception):
    pass

def create_solver_params(train_net_path, test_net_path=None, test_interv=1000, test_iter=100, base_lr=0.01,
    max_iter=40000, stepsize=10000):
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
    
    # Specify locations of the train and (maybe) test networks.
    solver.train_net = train_net_path
    if test_net_path is not None:
        solver.test_net.append(test_net_path)
        solver.test_interval = test_interv 
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
    solver.lr_policy = 'step'
    solver.gamma = 0.05
    solver.stepsize = stepsize
    
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
    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(solver))
        return f.name
    
    
def train_net(train_netspec, test_netspec=None, test_interv=1000, test_iter=100, base_lr=0.01,
            max_iter=40000, stepsize=10000, loss_blobs=None, acc_blobs=None,
            pretrained_weights="", snapshot=1000, snapshot_prefix="/tmp/snap"):
    """
    Run solvers for solver.max_iter iterations,
    returning the loss and accuracy recorded each iteration.

    :param train_netspec: Caffe NetSpec. The train network
    :param test_netspec: Caffe NetSpec. The test network 
    :param test_interv: int. Test will be performed ever test_interv iterations. Only useful if test_net_path is passed.
    :param base_lr: float. Initial learning rate
    :param max_iter: int. Maximum amount of iterations for training
    :param stepsize:  int. Amount of iterations between each lr update.
    :param loss_blobs: tuple of str. The names of the loss layers used to gather the loss info during training. These losses will then be returned as a result
    :param acc_blobs: tuple of str. Same as loss_blobs but with accuracies 
    :param pretrained_weights: str. Path where the pretrained .caffemodel is. If not set, train from scratch is performed.
    :param snapshot: int. Save snapshots (.solverstate and .caffemodel files) every 'snapshot' iterations.
    :param snapshot_prefix: str. Where to save the snapshots.
    :returns: str. A dict containing at most 3 keys:
              {
                'snaps': ['path/to/snapshot1', 'path/to/snapshot2', ... ],
                'loss': {
                           'loss_x': list containing the loss_x captured in every iteration,
                           'loss_y': list ...
                        },
                 'acc': {
                           'acc_x': list containing some accuracy captured every 'test_inverv' iterations,
                           ...
                        } 
              }
    """
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(train_netspec.to_proto()))
        train_net_path = f.name

    test_net_path = None
    if test_netspec:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(str(test_netspec.to_proto()))
            test_net_path = f.name

    solver_param = create_solver_params(train_net_path,
            test_net_path=test_net_path, test_interv=test_interv, test_iter=test_iter,
            base_lr=base_lr, max_iter=max_iter, stepsize=stepsize)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(solver_param))
        solver_param = f.name

    solver = caffe.get_solver(solver_param)

    if pretrained_weights and not exists(pretrained_weights):
        raise TrainException("Could not find pretrained weights with provided path: {}".format(pretrained_weights)) 
    elif exists(pretrained_weights):
        print("Loading weights from {}".format(pretrained_weights))
        solver.net.copy_from(pretrained_weights)

    results = {'snaps': []} 

    if loss_blobs:
        results['loss'] = {loss_name: np.array([]) for loss_name in loss_blobs}
    else:
        results['loss'] = {'loss': np.array([])}

    if not acc_blobs:
        acc_blobs = []
    if test_net_path and acc_blobs:
        results['acc'] = {} 

    if not exists(dirname(snapshot_prefix)):
        print("Path for snapshots does not exist. Creating dir {}".format(dirname(snapshot_prefix)))
        makedirs(dirname(snapshot_prefix))
    
    try:
        for it in range(0, max_iter+1):
            solver.step(1)
            # Retrieve loss of this step
            for name in loss_blobs:
                results['loss'][name] = np.append(results['loss'][name], solver.net.blobs[name].data.item())
            # Retrieve accuracy of tests every 'test_interv' iterations                            
            if it!=0 and it % test_interv == 0:
                for name in acc_blobs:     
                    results['acc'][name] = np.append(results['acc'].get(name, np.array([])), solver.test_nets[0].blobs[name].data.item())
            # Save snapshot every 'snapshot' iterations
            if it!=0 and it % snapshot == 0:
                snapshot_name = snapshot_prefix + '_iter_{}.caffemodel'.format(it)
                print("Saving snapshot in {}".format(snapshot_name))
                solver.net.save(snapshot_name)
                results['snaps'].append(snapshot_name)
    except KeyboardInterrupt:
        exit("Training has been interrupted. Bye!")

    return results 
