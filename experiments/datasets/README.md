## Datasets utils

Here you can find all the tools used to download and preprocess the datasets for all the experiments.

### Downloading the original datasets

- To download the MNIST dataset: `./download_mnist.sh`

- To download the KITTI dataset follow this [link](http://www.cvlibs.net/download.php?file=data_odometry_color.zip). For more info about the dataset visit the official [website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) 
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

- `preprocess_mnist_siamese`, which creates 2 databases for use with siamese networks: one LMDB contains the images and the other contains the labels for egomotion 

- `preprocess_mnist_standar`, which creates several databases to use in the finetuning steps of the siamese models. It also creates a test database with the 10K test images of MNIST 

