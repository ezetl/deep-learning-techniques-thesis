from os import makedirs
from os.path import join, exists, dirname
from sys import exit
import tempfile
import numpy as np
import caffe
from caffe.proto import caffe_pb2

class TrainException(Exception):
    pass

def create_solver_params(train_net_path, test_net_path=None, base_lr=0.01,
    max_iter=40000, stepsize=10000):
    """
    Code to create a solver. Taken and modified from
    this Caffe tutorial: https://github.com/BVLC/caffe/blob/master/examples/02-fine-tuning.ipynb
    :param train_net_path: str. Path to the train protoxt
    :param test_net_path: str. Path to the test protoxt
    :param base_lr: float. Initial learning rate
    """
    solver = caffe_pb2.SolverParameter()
    
    # Specify locations of the train and (maybe) test networks.
    solver.train_net = train_net_path
    if test_net_path is not None:
        solver.test_net.append(test_net_path)
        solver.test_interval = 1000
        solver.test_iter.append(100)
    
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
    
    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(solver))
        return f.name
    
    
def train_net(train_net_path, test_net_path=None, base_lr=0.01,
            max_iter=40000, stepsize=10000, loss_blobs=None,
            pretrained_weights="", snapshot=1000, snapshot_prefix="/tmp/snap"):
    """
    Run solvers for solver.max_iter iterations,
    returning the loss and accuracy recorded each iteration.

    :param solver: Caffe SolverParameter 
    :param max_iter: int. total number of iterations to run Solver
    """
    solver_param_name = create_solver_params(train_net_path, test_net_path,
            base_lr=base_lr, max_iter=max_iter, stepsize=stepsize)

    solver = caffe.get_solver(solver_param_name)

    if pretrained_weights:
        solver.net.copy_from(pretrained_weights)

    snapshots_list = []
    if loss_blobs:
        loss = {loss_name: np.zeros(max_iter) for loss_name in loss_blobs}
    else:
        loss = {'loss': np.zeros(max_iter)}

    if pretrained_weights and not exists(pretrained_weights):
        raise TrainException("Could not find pretrained weights with provided path: {}".format(pretrained_weights)) 

    if not exists(dirname(snapshot_prefix)):
        print("Path for snapshots does not exists. Creating dir {}".format(dirname(snapshot_prefix)))
        makedirs(dirname(snapshot_prefix))
    
    try:
        for it in range(max_iter):
            solver.step(1)
            for name in loss_blobs:
                loss[name][it] = solver.net.blobs[name].data.copy()
            if it != 0 and it % snapshot==0:
                snapshot_name = snapshot_prefix + '_iter_{}.caffemodel'.format(it)
                print("Saving snapshot in {}".format(snapshot_name))
                solver.net.save(snapshot_name)
                snapshots_list.append(snapshot_name)
    except KeyboardInterrupt:
        exit("Training has been interrupted. Bye!")

    return loss, snapshots_list
