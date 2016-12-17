from sys import exit
import tempfile
import numpy as np
import caffe
from caffe.proto import caffe_pb2


def create_solver_settings(train_net_path, test_net_path=None, base_lr=0.01,
        max_iter=40000, stepsize=10000, snapshot_prefix=""):
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

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    solver.snapshot = 1000
    solver.snapshot_prefix = snapshot_prefix 

    # Train on the GPU.
    solver.solver_mode = caffe_pb2.SolverParameter.GPU

    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(solver))
        return f.name


def run_solver(solver_path, max_iters=40000):
    """
    Run solvers for solver.max_iter iterations,
    returning the loss and accuracy recorded each iteration.

    :param solver: Caffe SolverParameter 
    :param max_iters: int. total number of iterations to run Solver
    """
    solver = caffe.get_solver(solver_path)
    blobs = ('loss_x', 'loss_y', 'loss_z')
    loss = {loss_name: np.zeros(max_iters) for loss_name in blobs}

    try:
        for it in range(max_iters):
            solver.step(1)
            for name in blobs:
                loss[name][it] = solver.net.blobs[name].data.copy()
        return loss
    except KeyboardInterrupt:
        exit("Training has been interrupted. Bye!")
