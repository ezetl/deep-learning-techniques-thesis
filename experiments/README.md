# Experiments 

## MNIST proof of concept experiments

0. Please read Section 3.4 of the [paper](https://arxiv.org/pdf/1505.01596v2.pdf) for further details of the experiment. The steps described here are more helpful if you understand what you are about to do :)

1. Go to `../datasets` and follow the [instructions](./datasets/README.md) to download the MNIST dataset and to create the LMDB. It's better if you create all the LMDBs in the same folder (the experiment script assumes that, otherwise you'll have to change some parameters) 

2. Once you have created the proper LMDBs for MNIST you can run the experiments by running the script `./experiment_mnist.py`. The script prints the accuracies at the end.


## KITTI experiment
0. Please read Section 4 of the [paper](https://arxiv.org/pdf/1505.01596v2.pdf) for further details of the experiment.

1. Go to `../datasets` and follow the [instructions](./datasets/README.md) to download the KITTI and SUN397 dataset and to create the LMDBs. It's better if you create all the LMDBs in the same folder (the experiment script assumes that, otherwise you'll have to change some parameters) 

2. Once you have created the proper LMDBs for KITTI you can run the experiments by running the script `./experiment_kitti.py`. The script prints the accuracies at the end. The accuracies reported correspond to a classification task using the SUN397 dataset by adding a classifier on top of each of the 5 convolutional layers of the CNN.

