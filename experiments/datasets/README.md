## Datasets utils

Here you can find all the tools used to download and preprocess the datasets for all the experiments.

### Downloading the original datasets

- To download the MNIST dataset: `./download_mnist.sh`

- To download the KITTI dataset follow this [link](http://www.cvlibs.net/download.php?file=data_odometry_color.zip). For more info about the dataset visit the official [website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) 

- To download SUN397 visit this [link](http://vision.princeton.edu/projects/2010/SUN/)

- To download the SF dataset, use the script `./download_sf.py` 

- To download ILSVRC'12 I suggest using the following [torrent](http://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2)


## Preprocessing the datasets and creating the LMDBs to train

LMDB is a database format widely used for deep learning given its performance with large amounts of data.

After you have downloaded the datasets, the next step is to preprocess the images and create the 
databases for the training of siamese networks and standard networks.

The preprocessing scripts were written in C++/Python, you can compile them like:

```
mkdir build; cd build/; cmake ..; make -j8
```

After compiling the C++ scripts, you'll find the tools to create the LMDBs inside the folder `build/tools` . Those are:

- `preprocess_mnist_siamese`, which creates 2 databases for use with siamese networks: one LMDB contains the images and the other contains the labels for egomotion. The data lmdb also contains the labels of SFA training. Execute the script without parameters to read the help message. 

- `preprocess_mnist_standar`, which creates several databases to use in the finetuning steps of the siamese models. It also creates a test database with the 10K test images of MNIST. Execute the script without parameters to read the help message 

- `preprocess_kitti_siamese`, which creates 2 databases (data and egomotion labels) for use with siamese networks in the KITTI experiment of the paper (Section 5.1 from the paper). The data lmdb also contains the labels of SFA training. Execute the script without parameters to read the help message.

- 1.`create_SUN_splits` 2.`preprocess_SUN` 3.`create_SUN_lmdbs` for the SUN397 dataset. First you should create the splits, then preprocess all the images and finally create the lmdbs. Read the scripts for further details about the parameters they take (or execute them without parameters and read the help message).

- 1.`create_ILSVRC_splits` 2.`create_ILSVRC_lmdbs`. Create the .txt files with the corresponding training/testing splits and then create the lmdbs using those. Execute the scripts without parameters to receive a help message.
