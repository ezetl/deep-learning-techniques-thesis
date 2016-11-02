# Introduction 

The files in this folder correspond to the experiments of the
Section 5.1 of the paper "Learning to see by moving" by Agrawal et al.

The experiments here assume that you already have the pretrained networks
KITTI-Net, KITTI-SFA-Net, SF-Net and SF-SFA-Net (read the paper to understand
how those networks are trained). To pretrain those models check the folders
experiments/{kitti,sf} respectively.

You should also have AlexNet's models pretrained with the dataset ILSVRC'12
(with 10, 20 and 1000 images per class). In this particular case we are going to
use AlexNet pretrained with 20 images per class (AlexNet-20K) and with 1000
images per class (AlexNet-1M). Check the folder experiments/alexnet to find how
to train those models.

# Steps

1. Once you have all the required pretrained models, you can download the SUN
   database from [here](http://vision.princeton.edu/projects/2010/SUN/).

2. Download the [Training and Testing partitions](http://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip) as well.

3. Create the list of (path, label) for each desired partition by running
   ./create_SUN_splits.py. Run the script once without arguments to see the help
   message.

4. Preprocess the images. It is important that the images have width == cols, so
   I made a small preprocessing script that crops the images by the smallest
   dimension.  Run ./preprocess_SUN.py once without parameters to see the help
   message.

5. Once you have the list of images and have preprocessed them, it's time to
   create a LMDB database.  There's a handy script in
   experiments/utils/create_lmdb.sh which takes a list of images/labels and creates
   a LMDB in a specified path by using Caffe tools. Run the script with the -h
   option to see the help message.

6. After you have created the LMDB, you can finetune the pretrained weights
   (Kitti, SF, AlexNet). In the folder ./prototxt you can find several networks
   models, each one corresponding to different locations of a classifier in the
   CNN (after the first convolutional layer, the second, etc.). Since the
   architecture of all the networks are based on AlexNet, you should be OK
   finetuning all the pretrained weights on any of these models.
   To finetune a pretrained weight run:
   ```
   cd network/model/you/want/to/train
   caffe train --gpu=all --solver=solver.prototxt --weights path/to/snapshot.caffemodel
   ```

7. (TODO) Once you have the finetuned weights, you can test how well they did by
   running the test/test.py script
