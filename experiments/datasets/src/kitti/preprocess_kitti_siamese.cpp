/*
 * This code preprocess the KITTI dataset.
 *
 * The main idea is to create a database with thousands of pairs of images to 
 * reproduce the results obtained in "Learning to See by Moving" by 
 * Agrawal el al.
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

#define THRESHOLD 8.8817841970012523e-16
#define NUM_BINS 20
#define HEIGHT 227
#define WIDTH 227
#define REAL_WIDTH 1241 
#define REAL_HEIGHT 376
#define NUM_CLASSES 3
#define LABEL_WIDTH NUM_BINS
#define NUM_CHANNELS 3
#define PAIRS_PER_SPLIT 2300 // approx. ~20K pairs of images
// Translation bins
// Maximum and minimum distances between pair of frames (rounded):
// maxx: 18 minx: -18
// maxz: 14 minz: -14
// maxy: 1.55734 miny: -1.53587
#define X_STEP 1.8
#define X_MIN -18
#define Z_STEP 1.3
#define Z_MIN -14
#define Y_MIN -0.563987
#define Y_STEP 0.0536 // approximate

#define PATHS_FILES  (DATA_ROOT"/kitti/paths/")
#define IMAGES       "/sequences/"
#define POSES        "/poses/"

#define LMDB_TRAIN      "kitti_train_egomotion_lmdb"
#define LMDB_LABEL_TRAIN "kitti_train_label_egomotion_lmdb"
#define LMDB_VAL        "kitti_val_egomotion_lmdb"
#define LMDB_LABEL_VAL "kitti_val_label_egomotion_lmdb"

typedef char Byte;
typedef unsigned char uByte;
typedef uByte Label;
typedef array< array<float, 4>, 3> TransformMatrix;
typedef array< array<float, 3>, 3> RotMatrix;
typedef struct
{
    string path1;
    TransformMatrix t1;
    int i1;
    string path2;
    TransformMatrix t2;
    int i2;
} ImgPair;
typedef struct 
{
    float x;
    float y;
    float z;
} EulerAngles;
typedef struct
{
    Mat img1;
    Mat img2;
    Label sfa;
    Label x;
    Label y;
    Label z;
} DataBlob;

// 9 Sequences for training, 2 for validation
const vector<string> TRAIN_SPLITS = {"00.txt", "01.txt", "02.txt", "03.txt", "04.txt", "05.txt", "06.txt", "07.txt", "08.txt"};
const vector<string> VAL_SPLITS = {"09.txt", "10.txt"};

unsigned int generate_rand(int range_limit);
RotMatrix multiply_rot_matrix(RotMatrix& t1, RotMatrix& t2);
RotMatrix get_rot_matrix(TransformMatrix& t);
EulerAngles mat2euler(RotMatrix& m);
void create_lmdbs(string images_root, string lmdb_path, const vector<string> split);
vector<ImgPair> generate_pairs(const string images_root, const vector<string> split);
DataBlob process_images(ImgPair p);

vector<ImgPair> generate_pairs(const string images_root, const vector<string> split) {
    vector<ImgPair> pairs_paths;
    for (unsigned int i=0; i<split.size(); ++i) {
        // Load original paths
        ifstream fsplit;
        fsplit.open(PATHS_FILES+split[i]);
        string path;
        vector<string> split_paths;
        while (fsplit >> path) {
            split_paths.push_back(images_root+"/"+IMAGES"/"+path);
        }
        fsplit.close();

        // Load transform matrix 
        fsplit.open(images_root+"/"+POSES+"/"+split[i]);
        TransformMatrix m;
        vector<TransformMatrix> split_matrix;
        while (fsplit >> m[0][0] >> m[0][1] >> m[0][2] >> m[0][3] >>
                         m[1][0] >> m[1][1] >> m[1][2] >> m[1][3] >> 
                         m[2][0] >> m[2][1] >> m[2][2] >> m[2][3]) {
            split_matrix.push_back(m);
        }

        // Generate pairs
        for (unsigned int j=0; j<PAIRS_PER_SPLIT; ++j) {
            int index = generate_rand(split_paths.size());
            int pair_index = 0;
            int pair_offset = generate_rand(7)+1;
            if (index==0) {
                pair_index = index + pair_offset;
            } else if (static_cast<unsigned int>(index) == split_paths.size()-1) {
                pair_index = index - pair_offset;
            } else {
                if (generate_rand(2)) { // go to the left
                    // Careful with this substraction. If the 2 operands were unsigned int we could get in trouble (overflow)
                    pair_index = (index - pair_offset >= 0) ? index - pair_offset : 0;  
                } else { // go to the right
                    pair_index = (static_cast<unsigned int>(index + pair_offset) <= split_paths.size()-1) ? index + pair_offset : split_paths.size()-1;  
                }
            }
            ImgPair pair = {split_paths[index], split_matrix[index], index, split_paths[pair_index], split_matrix[pair_index], pair_index};
            pairs_paths.push_back(pair);
        }
    }
    return pairs_paths;
}

void create_lmdbs(string images_root, string lmdb_path, const vector<string> split)
{
    string labels_path = lmdb_path + "_labels";
    LMDataBase *labels_lmdb = new LMDataBase(labels_path, (size_t)NUM_CLASSES, 1);
    LMDataBase *data_lmdb = new LMDataBase(lmdb_path, (size_t)6, (size_t)HEIGHT);

    // Generate pairs of images for each sequence 
    vector<ImgPair> pairs = generate_pairs(images_root, split);
    random_shuffle(std::begin(pairs), std::end(pairs));

    for (unsigned int i = 0; i<pairs.size(); i++)
    {
      DataBlob data = process_images(pairs[i]);
      data_lmdb->insert2db(data.img1, data.img2, data.sfa);
      vector<Label> labels = {(Label)data.x, (Label)data.y, (Label)data.z};
      labels_lmdb->insert2db(labels);
    }

    delete labels_lmdb;
    delete data_lmdb;
    return;
}

//float maxy=-40000.0, miny=40000.0;
DataBlob process_images(ImgPair p)
{
    DataBlob final_data;

    Mat im1 = imread(p.path1, CV_LOAD_IMAGE_COLOR);
    Mat im2 = imread(p.path2, CV_LOAD_IMAGE_COLOR);
    assert(im1.cols>0 && im2.cols>0 && im1.rows>0 && im2.rows>0);

    unsigned int top = generate_rand(min(im1.rows, im2.rows) - HEIGHT);
    unsigned int left = generate_rand(min(im1.cols, im2.cols) - WIDTH);
    Rect r(left, top, WIDTH, HEIGHT);

    final_data.img1 = im1(r);
    final_data.img2 = im2(r);

    float x,y,z;
    int bin_x = 0, bin_y = 0, bin_z = 0;
    // Translations
    x = p.t2[0][3] - p.t1[0][3];
    z = p.t2[2][3] - p.t1[2][3];

    // bin for x
    float base_x = X_MIN;
    while ((base_x += X_STEP) < x && bin_x < NUM_BINS) {
        ++bin_x;
    }
    // bin for z
    float base_z = Z_MIN;
    while ((base_z += Z_STEP) < z && bin_z < NUM_BINS) {
        ++bin_z;
    }

    // Euler angle
    //cout << "Transform matrix 1" << endl;
    //cout <<  "[[ " <<p.t1[0][0] << ", " <<  p.t1[0][1] << ", " <<  p.t1[0][2] << ", " <<  p.t1[0][3] << "]," << 
    //         "[ " <<p.t1[1][0] << ", " <<  p.t1[1][1] << ", " <<  p.t1[1][2] << ", " <<  p.t1[1][3] << "], " <<  
    //         "[ " <<p.t1[2][0] << ", " <<  p.t1[2][1] << ", " <<  p.t1[2][2] << ", " <<  p.t1[2][3] << "]]"<< endl;
    //cout << "Transform matrix 2" << endl;
    //cout <<  "[[ " <<p.t2[0][0] << ", " <<  p.t2[0][1] << ", " <<  p.t2[0][2] << ", " <<  p.t2[0][3] << "]," << 
    //         "[ " <<p.t2[1][0] << ", " <<  p.t2[1][1] << ", " <<  p.t2[1][2] << ", " <<  p.t2[1][3] << "]," <<  
    //         "[ " <<p.t2[2][0] << ", " <<  p.t2[2][1] << ", " <<  p.t2[2][2] << ", " <<  p.t2[2][3] << "]]" << endl;

    RotMatrix r1 = get_rot_matrix(p.t1);
    RotMatrix r2 = get_rot_matrix(p.t2);
    RotMatrix rot = multiply_rot_matrix(r1, r2);
    EulerAngles eu = mat2euler(rot);
    y = eu.y;
    //cout << "Final results:" << endl;
    //cout << "x trans: " << x << " Y euler: " << y << " Z trans: " << z << endl;
    //maxy = (y>maxy) ? y : maxy;
    //miny = (y<miny) ? y : miny;
    //std::cout << "maxy: " << maxy << " miny: " << miny << std::endl;
    // bin for y
    float base_y = Y_MIN;
    while ((base_y += Y_STEP) < y && bin_y < NUM_BINS) {
        ++bin_y;
    }

    final_data.x = bin_x;
    final_data.y = bin_y;
    final_data.z = bin_z;
    final_data.sfa = abs(p.i1 - p.i2) <= 7;

    // Debugging
    //namedWindow("im1");
    //namedWindow("im2");
    //imshow("im1", final_data.img1);
    //imshow("im2", final_data.img2);
    //waitKey(0);
    return final_data;
}

/* Generate a random number between 0 and range_limit-1
 * Useful to get a random element in an array of size range_limit
 */
unsigned int generate_rand(int range_limit)
{
    return rand() % range_limit;
}

RotMatrix multiply_rot_matrix(RotMatrix& t1, RotMatrix& t2){
    cv::Mat cvt1(3,3,CV_32FC1,&t1);
    cv::Mat cvt2(3,3,CV_32FC1,&t2);
    cv::Mat cvres = cvt2.t() * cvt1;
    RotMatrix res;
    for (int i = 0; i < cvres.rows; i++) {
      const float *cvr = cvres.ptr<float>(i);
      for (int j = 0; j < cvres.cols; j++) {
        res[i][j] = cvr[j];
      }
    }
    return res;
}

RotMatrix get_rot_matrix(TransformMatrix& t){
    RotMatrix rot;
    for (unsigned int i = 0; i<t.size(); ++i){
        for (unsigned int j = 0; j<t.size(); ++j){ // t is square
            rot[i][j] = t[i][j];
        }
    }
    return rot;
}

EulerAngles mat2euler(RotMatrix& m){
    float threshold = THRESHOLD;
	float cy = sqrt(m[2][2]*m[2][2] + m[1][2]*m[1][2]);
    float z=0.0, y=0.0, x=0.0;
    if (cy > threshold) {
        z = atan2(-m[0][1], m[0][0]);
        y = atan2(m[0][2], cy);
        x = atan2(-m[1][2], m[2][2]);
    } else {
        z = atan2(-m[1][0], m[1][1]);
        y = atan2(m[0][2], cy);
        x = 0.0;
    }
    //cout << "Rotmat" << endl;
    //cout <<  "[[ " <<m[0][0] << ", " <<  m[0][1] << ", " <<  m[0][2] << "]," << 
    //         "[ " <<m[1][0] << ", " <<  m[1][1] << ", " <<  m[1][2]  << "]," <<  
    //         "[ " <<m[2][0] << ", " <<  m[2][1] << ", " <<  m[2][2]  << "]]" << endl;
    //cout << "Euler:" << endl;
    //cout << "x: " << x << " y: " << y << " z: " << z << endl;
    return (EulerAngles){x,y,z};
}


int main(int argc, char** argv)
{
  if (argc < 3) {
    cout << "You must provide the path where the KITTI original dataset\n"
         << "lives ('sequences' and 'poses' folders downloaded from the official website)\n"
         << "and the path were you want to save your generated LMDBs:\n\n"
         << argv[0] << " path/to/sequences_and_poses path/where/to/save/LMDB\n\n";
  } else {
    srand(0);
    string lmdb_data_path = string(argv[2]) + "/" + LMDB_TRAIN;
    string images_root(argv[1]);
    cout << "Creating train LMDB's\n";
    create_lmdbs(images_root, lmdb_data_path, TRAIN_SPLITS);
    cout << "Creating val LMDB's\n";
    lmdb_data_path = string(argv[2]) + "/" + LMDB_VAL;
    create_lmdbs(images_root, lmdb_data_path, VAL_SPLITS);
  }
  return 0;
}
