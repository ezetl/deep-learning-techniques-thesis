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

// TODO(ezetl): use lmdb_creator here. wont compile otherwise
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"


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

#define PATHS_FILES  (DATA_ROOT"paths/")
#define IMAGES       "/media/eze/Datasets/KITTI/dataset/sequences/"
#define POSES        "/media/eze/Datasets/KITTI/dataset/poses/"

#define LMDB_ROOT       "/media/eze/Datasets/KITTI/" 
#define LMDB_TRAIN      (LMDB_ROOT"kitti_train_egomotion_lmdb/")
#define LMDB_LABEL_TRAIN (LMDB_ROOT"kitti_train_label_egomotion_lmdb")
#define LMDB_VAL        (LMDB_ROOT"kitti_val_egomotion_lmdb/")
#define LMDB_LABEL_VAL (LMDB_ROOT"kitti_val_label_egomotion_lmdb")

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
void create_lmdbs(const char* images, const char* lmdb_path, const char* lmdb_label_path, const vector<string> split);
vector<ImgPair> generate_pairs(const vector<string> split);
DataBlob process_images(ImgPair p);
void CVMatsToDatum(const Mat& img1, const Mat& img2, Datum* datum);

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

void create_lmdbs(const char* images, const char* lmdb_path, const char* lmdb_label_path, const vector<string> split)
{
    /***********************************************/
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
    /***********************************************/


    /***********************************************/
    // lmdb labels 
    MDB_env *mdb_envl;
    MDB_dbi mdb_dbil;
    MDB_val mdb_keyl, mdb_datal;
    MDB_txn *mdb_txnl;
    // Set database environment
    mkdir(lmdb_label_path, 0744);
    // Create Data LMDBs
    mdb_env_create(&mdb_envl);
    mdb_env_set_mapsize(mdb_envl, TB);
    mdb_env_open(mdb_envl, lmdb_label_path, 0, 0664);
    mdb_txn_begin(mdb_envl, NULL, 0, &mdb_txnl);
    mdb_open(mdb_txnl, NULL, 0, &mdb_dbil);
    /***********************************************/



    // Generate pairs of images for each sequence 
    vector<ImgPair> pairs = generate_pairs(split);
    random_shuffle(std::begin(pairs), std::end(pairs));

    // Dimensions of Data LMDB 
    unsigned int rows = HEIGHT;
    unsigned int cols = WIDTH;

    int count = 0;
    string data_value, label_value;
    
    /***********************************************/
    // Data datum
    // TODO (ezetlopez): update this comment if the changes work.
    // This datum has 3 + 3 + NUM_CLASSES dimensions.
    // The first 3 dimensions correspond to the first image, the second 3 to the second image 
    // respectively. NUM_CLASSES (3) because we are also merging the labels into the data.
    // This allow us to slice the data later on training phase and retrieve the labels.
    // Basically I am adding 3 "images" with all zeros in it except the index where the class 
    // is active. Then with the Argmax Layer of Caffe I retrieve these labels index again and pass them 
    // to the loss function.
    // Chek the loop below to see how it is done.
    Datum datum;
    //datum.set_channels(6+NUM_CLASSES);
    datum.set_channels(NUM_CHANNELS);
    datum.set_height(rows);
    datum.set_width(cols);
    /***********************************************/

    /***********************************************/
    // Labels datum
    Datum datuml;
    datuml.set_channels(NUM_CLASSES);
    datuml.set_height(1);
    datuml.set_width(1);
    /***********************************************/

    std::ostringstream s;

    for (unsigned int i = 0; i<pairs.size(); i++)
    {
        DataBlob data = process_images(pairs[i]);
        s << std::setw(8) << std::setfill('0') << count; 
        string key_str = s.str();
        s.str(std::string());

        /***********************************************/
        // Set Data
        assert (data.img1.isContinuous());
        assert (data.img2.isContinuous());
        CVMatsToDatum(data.img1, data.img2, &datum);
        datum.SerializeToString(&data_value);
        // Save Data
        mdb_data.mv_size = data_value.size();
        mdb_data.mv_data = reinterpret_cast<void*>(&data_value[0]);
        mdb_key.mv_size = key_str.size();
        mdb_key.mv_data = reinterpret_cast<void*>(&key_str[0]);
        mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0);
        /***********************************************/

        /***********************************************/
        // Set Labels 
        // Create a char pointer and copy the images first, the labels at the end
        char * labels = (char*) calloc(NUM_CLASSES, sizeof(uByte));
        labels[0] = (char)data.x;
        labels[1] = (char)data.y;
        labels[2] = (char)data.z;
        datuml.set_data((char*)labels, NUM_CLASSES);
        datuml.SerializeToString(&label_value);
        free(labels);
        // Save Labels
        mdb_datal.mv_size = label_value.size();
        mdb_datal.mv_data = reinterpret_cast<void*>(&label_value[0]);
        mdb_keyl.mv_size = key_str.size();
        mdb_keyl.mv_data = reinterpret_cast<void*>(&key_str[0]);
        mdb_put(mdb_txnl, mdb_dbil, &mdb_keyl, &mdb_datal, 0);
        /***********************************************/

        if (++count % 1000 == 0) {
            // Commit txn Data
            mdb_txn_commit(mdb_txn);
            mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn);
            // Commit txn labels 
            mdb_txn_commit(mdb_txnl);
            mdb_txn_begin(mdb_envl, NULL, 0, &mdb_txnl);
            cout << "Processed " << count << "\r" << flush;
        }
    }
    // Last batch
    if (count % 1000 != 0) {
        mdb_txn_commit(mdb_txn);
        mdb_close(mdb_env, mdb_dbi);
        mdb_env_close(mdb_env);

        mdb_txn_commit(mdb_txnl);
        mdb_close(mdb_envl, mdb_dbil);
        mdb_env_close(mdb_envl);
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
    srand(0);
    cout << "Creating train LMDB's\n";
    create_lmdbs(IMAGES, LMDB_TRAIN, LMDB_LABEL_TRAIN, TRAIN_SPLITS);
    cout << "Creating val LMDB's\n";
    create_lmdbs(IMAGES, LMDB_VAL, LMDB_LABEL_VAL, VAL_SPLITS);
    return 0;
}

void CVMatsToDatum(const Mat& img1, const Mat& img2, Datum* datum) {
    // Modified from CVMatToDatum from Caffe
    CHECK(img1.depth() == CV_8U) << "Image data type must be unsigned byte";
    CHECK(img2.depth() == CV_8U) << "Image data type must be unsigned byte";
    datum->set_channels(img1.channels() + img2.channels());
    datum->set_height(img1.rows);
    datum->set_width(img1.cols);
    datum->clear_data();
    datum->clear_float_data();
    datum->set_encoded(false);
    int datum_channels = datum->channels();
    int datum_height = datum->height();
    int datum_width = datum->width();
    int datum_size = datum_channels * datum_height * datum_width;
    string buffer(datum_size, ' ');

    for (int h = 0; h < datum_height; ++h) {
        const uchar* ptr1 = img1.ptr<uchar>(h);
        const uchar* ptr2 = img2.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < datum_width; ++w) {
            for (int c = 0; c < datum_channels/2; ++c) {
                int datum_index = (c * datum_height + h) * datum_width + w;
                buffer[datum_index] = static_cast<char>(ptr1[img_index]);
                datum_index = ((c+datum_channels/2) * datum_height + h) * datum_width + w;
                buffer[datum_index] = static_cast<char>(ptr2[img_index++]);
            }
        }
    }
    datum->set_data(buffer);
}
