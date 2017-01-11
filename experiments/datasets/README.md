## Datasets utils

Here you can find all the tools used to download and preprocess the datasets for all the experiments.

### Downloading the original datasets

- To download the MNIST dataset: `./download_mnist.sh`

- To download the KITTI dataset follow this [link](http://www.cvlibs.net/download.php?file=data_odometry_color.zip). For more info about the dataset visit the official [website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) 

- To download SUN397 visit this [link](http://vision.princeton.edu/projects/2010/SUN/)

- To download the SF dataset, use the script `./download_sf.py` 


## Preprocessing the datasets and creating the LMDBs to train

LMDB is a database format widely used for deep learning given its performance with large amounts of data.

After you have downloaded the datasets, the next step is to preprocess the images and create the 
databases for the training of siamese networks and standard networks.

The preprocessing code was written in C++, you can compile it like:

```
mkdir build; cd build/; cmake ..; make -j8
```

Inside the folder `build/tools` you'll find the compiled tools to create the LMDBs. Those are:

- `preprocess_mnist_siamese`, which creates 2 databases for use with siamese networks: one LMDB contains the images and the other contains the labels for egomotion. The data lmdb also contains the labels of SFA training. 

- `preprocess_mnist_standar`, which creates several databases to use in the finetuning steps of the siamese models. It also creates a test database with the 10K test images of MNIST 

- `preprocess_kitti_siamese`, which creates 2 databases (data and egomotion labels) for use with siamese networks in the KITTI experiment of the paper (Section 5.1 from the paper). The data lmdb also contains the labels of SFA training.

- `create_SUN_splits` `preprocess_SUN` `create_SUN_lmdbs` for the SUN397 dataset. First you should create the splits, then preprocess all the images and finally create the lmdbs by running those scripts in order. Read them for further details about the parameters they take.
