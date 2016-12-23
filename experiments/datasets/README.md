## Datasets utils
Here you can find all the tools used to download and preprocess the datasets for all the experiments.

### Downloading the original datasets
To download the MNIST dataset: `./download_mnist.sh`
To download the KITTI dataset follow this [link](http://www.cvlibs.net/download.php?file=data_odometry_color.zip). For more info about the dataset visit the official [website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) 


## Preprocessing the datasets and creating the LMDBs to train
LMDB is a database format widely used for deep learning given its performance with large amounts of data.
After you have downloaded the datasets, the next step is to preprocess the images and create the 
databases for the training of siamese networks and standard networks.

The preprocessing code was written in C++, you can compile it like:
```
mkdir build; cd build/; cmake ..; make -j8
```


