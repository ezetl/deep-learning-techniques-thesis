#!/usr/bin/env python2.7
import sys
import numpy as np
import matplotlib.pyplot as plt

"""
Script used to plot loss and accuracy from a Caffe log.
Author: Ezequiel Torti Lopez

## GENERAL NOTES:
Note that if you have several accuracies (i.e: train/test, top1/top5)
you should rename your output layers (Accuracy and Loss)
in different ways and then modify the variables 'acc_keywords' and
'loss_keywords' to parse these outputs properly.

'iter_keyword' should be a piece of text that appears in the line with the
number of iterations before testing. Usually in Caffe this line looks like:

  I0425 12:20:22.165951 17360 solver.cpp:341] Iteration 7000, Testing net (#0)

So I decided to put "Testing net" as 'iter_keyword'.

## MODIFYING OUTPUT LAYERS
For example, if you are computing the accuracy with the training set and the
validation set, you should rename your Accuracy layers with "train_accuracy"
and "val_accuracy", so the log would look like:

                                   .
                                   .
                                   .
solver.cpp:273] Snapshotting solver state ....
er.cpp:341] Iteration 7000, Testing net (#0)
cpp:748] Ignoring source layer data_train
er.cpp:409]     Test net output #0: train_accuracy = 0.87828
er.cpp:409]     Test net output #1: train_loss = 0.409493 (* 1 = 0.409493 loss)
er.cpp:341] Iteration 7000, Testing net (#1)
cpp:748] Ignoring source layer data_train
er.cpp:409]     Test net output #0: val_accuracy = 0.67248
er.cpp:409]     Test net output #1: val_loss = 1.06007 (* 1 = 1.06007 loss)
er.cpp:237] Iteration 7000, loss = 0.321616
                                   .
                                   .
                                   .

And then add the keywords:

acc_keywords = ['train_accuracy', 'val_accuracy']
loss_keywords = ['train_loss', 'val_loss']


## OUTPUT
The script saves a .png with the plots and show them on screen.


## HOW TO MODIFY YOUR .prototxt's: You can modify the last layers like this:
name: "CaffeNet"

layers {
    name: "data_train"
    type: DATA
    top: "data"
    top: "label"
    include {
        phase: TRAIN
    }
    transform_param {
        .
        .
    }
}
   .
   .
   .
layers {
  name: "val_accuracy"
  type: ACCURACY
  bottom: "fc8"
  bottom: "label"
  top: "val_accuracy"
  include {
      phase: TEST
      stage: "test-on-test"
   }
}
layers {
  name: "train_accuracy"
  type: ACCURACY
  bottom: "fc8"
  bottom: "label"
  top: "train_accuracy"
  include {
      phase: TEST
      stage: "test-on-train"
    }
}
layers {
  name: "loss"
  type: SOFTMAX_LOSS
  bottom: "fc8"
  bottom: "label"
  top: "loss"
  include {
      phase: TRAIN
    }
}
layers {
  name: "train_loss"
  type: SOFTMAX_LOSS
  bottom: "fc8"
  bottom: "label"
  top: "train_loss"
  include {
      phase: TEST
      stage: "test-on-train"
    }
}
layers {
  name: "test_loss"
  type: SOFTMAX_LOSS
  bottom: "fc8"
  bottom: "label"
  top: "test_loss"
  include {
      phase: TEST
      stage: "test-on-test"
    }
}
"""

# This word appears in the same line where I get the iteration number.
# So I use it to 'grep' the line
iter_keyword = 'Testing net'

# Try to maintain the order of the data throughout '{acc,loss}_keywords'
# (i.e. try to use the same position of each array for the data related to the
# same stage during training)
# Example: acc_keywords[0]  = top1_accuracy, acc_keywords[1]  = top5_accuracy
#          loss_keyworks[0] = top1_loss,     loss_keyworks[1] = top5_loss
acc_keywords = ['train_accuracy', 'test_accuracy']
loss_keywords = ['train_loss', 'test_loss']

# Do not grep the lines with these words
forbbiden = ['NetState', 'name', 'top', 'Creating', '<-', '->', 'Setting up',
             'backward computation', 'produces output']

# Line colors for matplotlib
colors = ['#1f77b4', '#bcbd22', '#ff7f0e', '#d62728', '#2ca02c']


def get_iter_num(line=""):
    num = line.split('Iteration')[1]
    num = num.split(',')[0].strip()
    return int(num)


def get_loss(line=""):
    loss = line.split('(')[0]
    loss = loss.split('=')[1].strip()
    return float(loss)


def get_accuracy(line=""):
    acc = line.split('=')[1].strip()
    return float(acc)


def check_empty(accuracies={}, losses={}, iters=[]):
    if len(iters) < 3:
        print("\nYou need more data to plot, try\n\
              \ragain later with more iterations.\n")
        sys.exit(1)

    some_empty = False
    for key in acc_keywords:
        if len(accuracies[key][1:]) == 0:
            some_empty = True
            print("\nThe list of accuracies corresponding to '{}' was empty.\n\
                  \rMaybe you have outdated/wrong keywords.\n"
                  .format(key))

    for key in loss_keywords:
        if len(losses[key][1:]) == 0:
            some_empty = True
            print("\nThe list of loss corresponding to '{}' was empty.\n\
                  \rMaybe you have outdated/wrong keywords.\n"
                  .format(key))

    if some_empty:
        sys.exit(1)


def parse_log(filename=""):
    f = open(filename, 'r')
    lines = f.read().splitlines()

    accuracies = {}
    for acc in acc_keywords:
        accuracies[acc] = np.zeros(1, dtype=float)

    losses = {}
    for loss in loss_keywords:
        losses[loss] = np.zeros(1, dtype=float)

    # Here we get iters, accuracies and loss
    iters = set()
    for line in lines:
        if iter_keyword in line:
            iters.add(get_iter_num(line))
    iters = list(iters)
    iters.sort()
    iters = np.array(iters)

    for lin in lines:
        for key in acc_keywords:
            if key in lin and not any(f in lin for f in forbbiden):
                accuracies[key] = np.append(accuracies[key], get_accuracy(lin))

    for lin in lines:
        for key in loss_keywords:
            if key in lin and not any(f in lin for f in forbbiden):
                losses[key] = np.append(losses[key], get_loss(lin))

    check_empty(accuracies, losses, iters)

    return (accuracies, losses, iters)


def plot_log(accuracies={}, losses={}, iters=[]):

    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.set_ylabel('accuracy')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('iterations')
    ax0.spines['top'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax0.get_xaxis().tick_bottom()
    ax0.get_yaxis().tick_left()
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    # Plot baseline: accuracy 100%
    ax0.plot((iters[0], iters[-1]), (1, 1), 'r--')
    # Trick to force the yticks up to 1.1
    ax0.plot((iters[0], iters[-1]), (1.1, 1.1), 'w-')

    # Plot the accuracies
    for i, k in enumerate(acc_keywords):
        if len(iters) != len(accuracies[k][1:]):
            print("\nThe amount of iterations and the corresponding accuracies\n\
                  \r{} have different lenghts ({} vs. {}). There might be\n\
                  \rsomething wrong with your log file (incomplete download)\n\
                  \rtry downloading it again\n".format(k,
                                                       len(iters),
                                                       len(accuracies[k][1:])))
            sys.exit(1)

        ax0.plot(iters, accuracies[k][1:], label=k,
                 color=colors[i % len(colors)])

    ax0.legend(loc=4)

    # Plot the loss
    for i, k in enumerate(loss_keywords):
        if len(iters) != len(losses[k][1:]):
            print("\nThe amount of iterations and the corresponding loss\n\
                  \r{} have different lenghts ({} vs. {}). There might be\n\
                  \rsomething wrong with your log file (incomplete download)\n\
                  \rtry downloading it again\n".format(k,
                                                       len(iters),
                                                       len(losses[k][1:])))
            sys.exit(1)

        ax1.plot(iters, losses[k][1:], label=k, color=colors[i % len(colors)])

    ax1.legend(loc=1)

    plt.show()
    return


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('\nYou have to provide the log file:')
        print('{} caffe.log\n'.format(sys.argv[0]))
        sys.exit(1)
    accuracies, losses, iters = parse_log(sys.argv[1])
    plot_log(accuracies, losses, iters)
    sys.exit(0)
