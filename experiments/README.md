# Experiments 

## MNIST proof of concept experiments

0. Please read Section 3.4 of the [paper](https://arxiv.org/pdf/1505.01596v2.pdf) for further details of the experiment. The steps described here are more helpful if you understand what you are about to do :)

1. Go to `../datasets` and follow the [instructions](./datasets/README.md) to download the MNIST dataset and to create the LMDB. It's better if you create all the LMDBs in the same folder (the experiment script assumes that, otherwise you'll have to change some parameters) 

2. Once you have created the proper LMDBs for MNIST you can run the experiments by running the script `./experiment_mnist.py`. The script prints the accuracies at the end.

3. Once you have created the proper LMDBs for KITTI and SUN397 and Imagenet using the tools in `dataset/build/tools`, you can run the experiments by running the script `./experiment_kitti.py`. The script prints the accuracies at the end.
