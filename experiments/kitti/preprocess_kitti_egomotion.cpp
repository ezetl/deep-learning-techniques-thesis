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

#include <iostream>
#include <fstream>
#include <string> 
#include <string.h>
#include <vector>
#include <array>
#include <cmath>
#include <tuple>
#include <cinttypes>
#include <sys/stat.h>
#include <stdlib.h> 
#include <iomanip>
#include <algorithm>

#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/proto/caffe.pb.h"


using namespace caffe;
using namespace std;
using namespace cv;

#define TB 1099511627776
#define THRESHOLD 8.8817841970012523e-16
#define NUM_BINS 20
#define HEIGHT 227
#define WIDTH 227
#define REAL_WIDTH 1241 
#define REAL_HEIGHT 376
#define NUM_CLASSES 3
#define LABEL_WIDTH NUM_BINS
#define PAIRS_PER_SPLIT  20000 
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

#define DATA_ROOT    "../data/"
#define PATHS_FILES  (DATA_ROOT"paths/")
#define IMAGES       "/media/eze/0F4A13791A35DD40/KITTI/dataset/sequences/"
#define POSES        "/media/eze/0F4A13791A35DD40/KITTI/dataset/poses/"

#define LMDB_ROOT       "/media/eze/0F4A13791A35DD40/KITTI/" 
#define LMDB_TRAIN      (LMDB_ROOT"kitti_train_egomotion_lmdb/")
#define LMDB_VAL        (LMDB_ROOT"kitti_val_egomotion_lmdb/")

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
void create_lmdbs(const char* images, const char* lmdb_path, const vector<string> split);
vector<ImgPair> generate_pairs(const vector<string> split);
DataBlob process_images(ImgPair p);

vector<ImgPair> generate_pairs(const vector<string> split) {
    vector<ImgPair> pairs_paths;
    for (unsigned int i=0; i<split.size(); ++i) {
        // Load original paths
        ifstream fsplit;
        fsplit.open(PATHS_FILES+split[i]);
        string path;
        vector<string> split_paths;
        while (fsplit >> path) {
            split_paths.push_back(IMAGES+path);
        }
        fsplit.close();

        // Load transform matrix 
        fsplit.open(POSES+split[i]);
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

void create_lmdbs(const char* images, const char* lmdb_path, const vector<string> split)
{
    // lmdb data 
    MDB_env *mdb_env;
    MDB_dbi mdb_dbi;
    MDB_val mdb_key, mdb_data;
    MDB_txn *mdb_txn;

    // Set database environment
    mkdir(lmdb_path, 0744);

    // Create Data LMDBs
    mdb_env_create(&mdb_env);
    mdb_env_set_mapsize(mdb_env, TB);
    mdb_env_open(mdb_env, lmdb_path, 0, 0664);
    mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn);
    mdb_open(mdb_txn, NULL, 0, &mdb_dbi);

    // Generate pairs of images for each sequence 
    vector<ImgPair> pairs = generate_pairs(split);
    random_shuffle(std::begin(pairs), std::end(pairs));

    // Dimensions of Data LMDB 
    unsigned int rows = HEIGHT;
    unsigned int cols = WIDTH;

    int count = 0;
    string data_value, label_value;
    
    // Data datum
    // This datum has 3 + 3 + NUM_CLASSES dimensions.
    // The first 3 dimensions correspond to the first image, the second 3 to the second image 
    // respectively. NUM_CLASSES (3) because we are also merging the labels into the data.
    // This allow us to slice the data later on training phase and retrieve the labels.
    // Basically I am adding 3 "images" with all zeros in it except the index where the class 
    // is active. Then with the Argmax Layer of Caffe I retrieve these labels index again and pass them 
    // to the loss function.
    // Chek the loop below to see how it is done.
    Datum datum;
    datum.set_channels(6+NUM_CLASSES);
    datum.set_height(rows);
    datum.set_width(cols);

    std::ostringstream s;

    for (unsigned int i = 0; i<pairs.size(); i++)
    {
        DataBlob data = process_images(pairs[i]);
        s << std::setw(8) << std::setfill('0') << count; 
        string key_str = s.str();
        s.str(std::string());

        // Set Data
        // Create a char pointer and copy the images first, the labels at the end
        char * data_label;
        data_label = (char*)calloc(rows*cols*(6+NUM_CLASSES), sizeof(uByte));
        data_label = (char*)memcpy(data_label, (void*)data.img1.data, 3*rows*cols);
        memcpy(data_label+(cols*rows*3), (void*)data.img2.data, 3*rows*cols);

        char * labels;
        unsigned int labelx = (unsigned int)data.x;
        unsigned int labely = (unsigned int)data.y;
        unsigned int labelz = (unsigned int)data.z;
        labels = (char*) calloc(rows*cols*3, sizeof(uByte));
        labels[labelx] = 1;
        labels[cols*rows + labely] = 1;
        labels[cols*rows*2 + labelz] = 1;
        memcpy(data_label+(cols*rows*6), (void*)labels, 3*rows*cols);

        datum.set_data((char*)data_label, (6+NUM_CLASSES)*rows*cols);
        datum.SerializeToString(&data_value);

        // Save Data
        mdb_data.mv_size = data_value.size();
        mdb_data.mv_data = reinterpret_cast<void*>(&data_value[0]);
        mdb_key.mv_size = key_str.size();
        mdb_key.mv_data = reinterpret_cast<void*>(&key_str[0]);
        mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0);

        if (++count % 1000 == 0) {
            // Commit txn Data
            mdb_txn_commit(mdb_txn);
            mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn);
            cout << "Processed " << count << "\r" << flush;
        }
        free(data_label);
        free(labels);
    }
    // Last batch
    if (count % 1000 != 0) {
        mdb_txn_commit(mdb_txn);
        mdb_close(mdb_env, mdb_dbi);
        mdb_env_close(mdb_env);
    }

    cout << "\nFinished creation of LMDB. " << count << " images processed.\n";
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
    RotMatrix r1 = get_rot_matrix(p.t1);
    RotMatrix r2 = get_rot_matrix(p.t2);
    RotMatrix rot = multiply_rot_matrix(r1, r2);
    EulerAngles eu = mat2euler(rot);
    y = eu.y;
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

    /*
    // Debugging
    namedWindow("im1");
    namedWindow("im2");
    imshow("im1", crop1);
    imshow("im2", crop2);
    waitKey(100);
    */
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
    //memset((void*)&res, 0, sizeof(RotMatrix));
    //unsigned int rot_size = t1.size();
    //for (unsigned int k=0; k<rot_size; ++k) {
    //    for (unsigned int i=0; i<rot_size; ++i){
    //        for (unsigned int j = 0; j<rot_size; ++j){
    //            res[i][k] += t1[i][j] * t2[j][k];
    //        }
    //    }
    //}
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
    return (EulerAngles){x,y,z};
}


int main(int argc, char** argv)
{
    srand(0);
    cout << "Creating train LMDB's\n";
    create_lmdbs(IMAGES, LMDB_TRAIN, TRAIN_SPLITS);
    cout << "Creating val LMDB's\n";
    create_lmdbs(IMAGES, LMDB_VAL, VAL_SPLITS);
    return 0;
}
