/*
 * This code parses the MNIST dataset files (images and labels).
 *
 * The main idea is to create a database with million of images to
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
 * so I will create a separate database with an array of labels for each
 * element.
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

#define NUM_TRASLATIONS 7
#define NUM_ROTATIONS 60
#define NUM_BIN_ROTATIONS 20
#define NUM_CLASSES 3
#define LABEL_WIDTH NUM_BIN_ROTATIONS
#define LOWER_ANGLE -31
#define LOWER_TRASLATION -3
#define BATCHES 6

typedef struct {
  Mat img1;
  Mat img2;
  Label x;
  Label y;
  Label z;
} DataBlob;

void create_lmdb(string images, string lmdb_path);
Mat transform_image(Mat &img, float tx, float ty, float rot);
vector<DataBlob> process_images(vector<Mat> &list_imgs, unsigned int pairs_per_img);
unsigned int generate_rand(int range_limit);

int main(int argc, char **argv) {
  if (argc < 3) {
    cout << "You must provide the path where the MNIST original dataset\n"
         << "lives and the path were you want to save your generated LMDBs:\n\n"
         << argv[0] << " path/to/train-images-idx3-ubyte path/where/to/save/LMDB\n\n";
    cout << "Please use the script experiments/mnist/download_mnist.sh to get the "
         << "original version of the MNIST dataset\n\n";
  } else {
    cout << "Creating LMDBs\n";
    string orig_imgs_path(argv[1]);
    string lmdb_data_path = string(argv[2]) + "/mnist_train_siamese_lmdb";
    create_lmdb(orig_imgs_path, lmdb_data_path);
  }
  return 0;
}

void create_lmdb(string images, string lmdb_path) {
  // Load images/labels
  vector<Mat> list_imgs = load_images(images);

  // Create databases objects
  string labels_path = labels_path + "_labels";
  LMDataBase *labels_lmdb = new LMDataBase(labels_path, (size_t)NUM_CLASSES, 1);
  LMDataBase *data_lmdb = new LMDataBase(lmdb_path, (size_t)2, (size_t)list_imgs[0].rows);

  // Processing and generating million of images at once will consume too much RAM (>7GB) and it will
  // (probably) throw a std::bad_alloc exception. Lets split the processing in several batches instead.
  // list_imgs.size() has to be multiple of BATCHES (to simplify things)
  int len_batch = list_imgs.size() / BATCHES;
  for (unsigned int i = 0; i < BATCHES; i++) {
    unsigned int begin = i * len_batch;
    unsigned int end = begin + len_batch;
    vector<Mat> batch_imgs = vector<Mat>(list_imgs.begin() + begin, list_imgs.begin() + end);
    unsigned int pairs_per_img = 83 * (i != 0) + 85 * (i == 0); // for a total of 5million imgs
    vector<DataBlob> batch_data = process_images(batch_imgs, pairs_per_img);
    cout << "Batch images: " << batch_imgs.size() << " Batch pairs: " << batch_data.size() << endl;
    random_shuffle(std::begin(batch_data), std::end(batch_data));
    for (unsigned int item_id = 0; item_id < batch_data.size(); ++item_id) {
      int sfa_label = (Label)(batch_data[item_id].x >= 2 && batch_data[item_id].x <= 4 && batch_data[item_id].y >= 2 &&
                              batch_data[item_id].y <= 4 && (batch_data[item_id].z == 9 || batch_data[item_id].z == 10));
      data_lmdb->insert2db(batch_data[item_id].img1, batch_data[item_id].img2, sfa_label);
      vector<Label> labels = {(Label)batch_data[item_id].x, (Label)batch_data[item_id].y,
                              (Label)batch_data[item_id].z};
      labels_lmdb->insert2db(labels);
    }
  }
  delete labels_lmdb;
  delete data_lmdb;
  return;
}

/*
 * rot (Rotation) is in degrees
 * tx, ty (Translations) are pixels
 */
Mat transform_image(Mat &img, float tx, float ty, float rot) {
  Mat res;
  Point2f mid(img.cols / 2, img.rows / 2);
  Mat rotMat = getRotationMatrix2D(mid, rot, 1.0);
  Mat transMat = (Mat_<double>(2, 3) << 0, 0, tx, 0, 0, ty);
  rotMat = rotMat + transMat;
  // Set constant value for border to be white
  warpAffine(img, res, rotMat, Size(img.cols, img.rows), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
  return res;
}

vector<DataBlob> process_images(vector<Mat> &list_imgs, unsigned int pairs_per_img) {
  vector<DataBlob> final_data;
  srand(0);
  unsigned int rand_index = 0;
  vector<float> translations(NUM_TRASLATIONS);
  float value = LOWER_TRASLATION;
  for (unsigned int i = 0; i < translations.size(); i++) {
    translations[i] = value++;
  }

  value = LOWER_ANGLE;
  vector<float> rotations(NUM_ROTATIONS);
  for (unsigned int i = 0; i < rotations.size(); i++) {
    rotations[i] = (++value == 0) ? ++value : value;
  }

  // Debugging
  // namedWindow("Normal");
  // namedWindow("Transformed");
  for (unsigned int i = 0; i < list_imgs.size(); i++) {
    for (unsigned int j = 0; j < pairs_per_img; j++) {
      DataBlob d;
      // Generate random X translation
      rand_index = generate_rand(NUM_TRASLATIONS);
      d.x = rand_index;
      float tx = translations[rand_index];
      // Generate random Y translation
      rand_index = generate_rand(NUM_TRASLATIONS);
      d.y = rand_index;
      float ty = translations[rand_index];
      // Calculate random bin of rotation (0 to 19)
      rand_index = generate_rand(NUM_BIN_ROTATIONS);
      d.z = rand_index;
      // Calculate the real index of the array of rotations (0 to 61)
      rand_index *= 3;
      rand_index += generate_rand(3);
      float rot = rotations[rand_index];

      // Finally, apply the selected transformations to the image
      Mat new_img = transform_image(list_imgs[i], tx, ty, rot);

      d.img1 = list_imgs[i];
      d.img2 = new_img;

      if (generate_rand(2)) {
        d.img1 = new_img;
        d.img2 = list_imgs[i];
      }

      final_data.push_back(d);

      // Debugging
      // imshow("Normal", list_imgs[i]);
      // imshow("Transformed", new_img);
      // waitKey(100);
    }
  }
  return final_data;
}

/* Generate a random number between 0 and range_limit-1
 * Useful to get a random element in an array of size range_limit
 */
unsigned int generate_rand(int range_limit) { return rand() % range_limit; }
