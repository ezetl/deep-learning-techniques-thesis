/*
 * This code parses the MNIST dataset files (images and labels).
 *
 * The main idea is to create a database with 5 million images to 
 * reproduce the results obtained in "Learning to See by Moving" by 
 * Agrawal el al.
 *
 * For that, I parse the MNIST data, apply the transformations mentioned
 * in the section 3.4.1 of the paper and save the images to a LMDB.
 *
 * Since Agrawal et al. pose the problem as a classification problem
 * (discrete clases for transformations in the axis X, Y and Z) we 
 * need three classifiers, one for each dimension; hence we need three 
 * labels. It is not straightforward to create a multiclass database in LMDB,
 * so I will create a separate database with an array of labels for each element.
 * Then, during training phase, I just Slice the label to retrieve the 
 * labels again. Check this post to get a more complete idea:
 * https://groups.google.com/forum/#!searchin/caffe-users/multilabel/caffe-users/RuT1TgwiRCo/hoUkZOeEDgAJ 
 *
 * This code is part of my undergrad thesis: "Reconocimiento visual
 * empleando técnicas de deep learning" ("Visual Recognition using Deep
 * Learning techniques")
 *
 * Author: Ezequiel Torti Lopez
 */

#include "lmdb_creator.hpp"
#include "mnist_utils.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include <cinttypes>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace std;
using namespace cv;

#define TB 1099511627776

#define TRAIN_IMAGES (DATA_ROOT"train-images-idx3-ubyte")
#define TRAIN_LABELS (DATA_ROOT"train-labels-idx1-ubyte")
#define TEST_IMAGES  (DATA_ROOT"t10k-images-idx3-ubyte")
#define TEST_LABELS  (DATA_ROOT"t10k-labels-idx1-ubyte")

#define LMDB_SIZE 10000 
#define LMDB_ROOT         "/media/eze/Datasets/MNIST/"
#define LMDB_TRAIN      (LMDB_ROOT"mnist_finetuning_standar10000_lmdb/")
#define LMDB_TEST       (LMDB_ROOT"mnist_test_standar_lmdb/")

void create_lmdbs(const char* images, const char* labels, const char* lmdb_path, unsigned int size);

int main(int argc, char** argv)
{
    cout << "Creating train LMDB\n";
    create_lmdbs(TRAIN_IMAGES, TRAIN_LABELS, LMDB_TRAIN, LMDB_SIZE);
    cout << "Creating test LMDB\n";
    create_lmdbs(TEST_IMAGES, TEST_LABELS, LMDB_TEST, 10000);
    return 0;
}

void create_lmdbs(const char* images, const char* labels, const char* lmdb_path, unsigned int size)
{

    // Load images/labels
    vector<Mat> list_imgs = load_images(images);
    vector<Label> list_labels = load_labels(labels);

    vector< pair<Mat, Label> > imgs_labels(list_imgs.size());
    for (unsigned int i = 0; i<list_imgs.size(); i++)
    {
        imgs_labels[i] = pair<Mat, Label>(list_imgs[i], list_labels[i]);

    }
    random_shuffle(std::begin(imgs_labels), std::end(imgs_labels));

    for (unsigned int i = 0; i<size; i++)
    {
    }

    cout << "\nFinished creation of LMDB's\n";
    return;
}
