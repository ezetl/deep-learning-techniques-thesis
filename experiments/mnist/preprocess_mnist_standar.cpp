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
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

#define TRAIN_IMAGES "/train-images-idx3-ubyte"
#define TRAIN_LABELS "/train-labels-idx1-ubyte"
#define TEST_IMAGES "/t10k-images-idx3-ubyte"
#define TEST_LABELS "/t10k-labels-idx1-ubyte"

const vector<unsigned int> sizes = {100, 300, 1000, 10000, 60000};
void create_lmdbs(string images, string labels, string lmdb_path, unsigned int size);

int main(int argc, char **argv) {
  if (argc < 3) {
    cout << "You must provide the path where the MNIST original files (train-images-idx3-ubyte, etc.)\n"
         << "live and the path were you want to save your generated LMDBs:\n\n"
         << argv[0] << " path/to/MNIST/files path/where/to/save/LMDB\n\n";
    cout << "Please use the script experiments/mnist/download_mnist.sh to get the "
         << "original version of the MNIST dataset\n\n";
  } else {
    string mnist_data_path(argv[1]);
    string lmdb_path(argv[2]);
    for (unsigned int i = 0; i < sizes.size(); ++i) {
      cout << "Creating train LMDB for size " << sizes[i] << endl;
      string complete_lmdb_path = lmdb_path + "/mnist_standar_lmdb_"+ to_string(sizes[i]);
      create_lmdbs(mnist_data_path + TRAIN_IMAGES,
                   mnist_data_path + TRAIN_LABELS,
                   lmdb_path + "/mnist_standar_lmdb_"+ to_string(sizes[i]),
                   sizes[i]);
    }
    cout << "Creating test LMDB\n";
    create_lmdbs(mnist_data_path + TEST_IMAGES, 
                 mnist_data_path + TEST_LABELS,
                 lmdb_path+"/mnist_standar_lmdb_test",
                 10000);
  }
  return 0;
}

void create_lmdbs(string images, string labels, string lmdb_path, unsigned int size) {

  // Load images/labels
  vector<Mat> list_imgs = load_images(images);
  vector<Label> list_labels = load_labels(labels);

  LMDataBase *data_lmdb = new LMDataBase(lmdb_path, (size_t)1, (size_t)list_imgs[0].rows);

  vector<pair<Mat, Label>> pairs_img_label(list_imgs.size());
  for (unsigned int i = 0; i < list_imgs.size(); i++) {
    pairs_img_label[i] = pair<Mat, Label>(list_imgs[i], list_labels[i]);
  }
  random_shuffle(std::begin(pairs_img_label), std::end(pairs_img_label));
  for (unsigned int i = 0; i < size; ++i) {
    data_lmdb->insert2db(pairs_img_label[i].first, static_cast<int>(pairs_img_label[i].second));
  }
  delete data_lmdb;

  cout << "\nFinished creation of LMDB's\n";
  return;
}
