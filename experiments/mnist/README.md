MNIST preliminar experiments
----------------------------

This folder contains all the code and networks models related to the MNIST experiment of the paper _Learning to See by Moving_ by Agrawal et al.


Contents:
--------

- `download_mnist.sh` : downloads the original MNIST images

- `prototxt/` : all the CNNs models for the siamese/standar/evaluation parts of the experiments lie here

- `src/` : all the code related to the preprocessing of the images and creation of LMDBS

- `test/` : code related to test the CNNs performance (accuracy/error rate) and plot loss/accuracy vs. number of iterations 


Steps to reproduce results:
---------------------------

0. Please read Section 3.4 of the [paper](https://arxiv.org/pdf/1505.01596v2.pdf) for further details of the experiment. The steps described here are more helpful if you understand what you are about to do :)

1. Download the original MNIST dataset by executing `download_mnist.sh`. The script also creates some extra folders for later use

2. `cd ../; mkdir build; cd build; cmake ..; make -j8` to compile the preprocessing scripts 

3. Prepare the LMDBs to train the siamese networks by executing the scripts you just compiled in `../build/mnist/preprocess_mnist_siamese` and `../build/mnist/preprocess_mnist_standar` 
   Execute them without any arguments to read the help message.

4. Once you have created the proper LMDBs you have to modify the .prototxt files in `./prototxt/` to point the input layers to your LMDBs. Please do read those files and change the `data_param.source` param of the `Data` layer to your recently created LMDB 

5. Train the CNN by stepping in the folder of the experiment you've chosen (`./prototxt/egomotion`, `./prototxt/contrastive`, `./prototxt/finetuning` or `./prototxt/standar`) and running the following command:

   ```
   caffe train --gpu=all --solver=solver.prototxt
   ```

   In my case, "caffe" is an alias to the tool located in `/home/eze/.Software/caffe/build/tools/caffe`.

6. Once you trained your network you can finetune the model in `./prototxt/finetuning`. Please also change the LMDB's paths to one of your LMDBs created using `../build/mnist/preprocess_mnist_standar`.

7. Evaluate the final model by running (also inside `./prototxt/finetuning`):

   ```
   caffe test --gpu=all --model=finetuning_mnist.prototxt --weights=../../snapshots/some.caffemodel --iterations 500
   ```
