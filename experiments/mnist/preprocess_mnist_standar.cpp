/*
 * This code parses the MNIST dataset files (images and labels).
 *
 * It was done for my low-endian machine, but you can set the LOW_ENDIAN
 * flag off and it will run in high endian mode (TODO)
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

#include <iostream>
#include <fstream>
#include <string> 
#include <string.h>
#include <vector>
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

#define LOW_ENDIAN true
#define TB 1099511627776

#define DATA_ROOT    "../data/"
#define TRAIN_IMAGES (DATA_ROOT"train-images-idx3-ubyte")
#define TRAIN_LABELS (DATA_ROOT"train-labels-idx1-ubyte")
#define TEST_IMAGES  (DATA_ROOT"t10k-images-idx3-ubyte")
#define TEST_LABELS  (DATA_ROOT"t10k-labels-idx1-ubyte")

#define LMDB_SIZE 300
#define LMDB_ROOT         "../data/"
#define LMDB_TRAIN        (LMDB_ROOT"mnist_train_standar_lmdb/")
#define LMDB_TEST       (LMDB_ROOT"mnist_test_standar_lmdb/")

typedef char Byte;
typedef unsigned char uByte;
typedef uByte Label;
typedef struct
{
    uint32_t magic;
    uint32_t num_elems;
    uint32_t cols;
    uint32_t rows;
} MNIST_metadata;

void create_lmdbs(const char* images, const char* labels, const char* lmdb_path);
uint32_t get_uint32_t(ifstream &f, streampos offset);
vector<uByte> read_block(ifstream &f, unsigned int size, streampos offset);
MNIST_metadata parse_images_header(ifstream &f);
MNIST_metadata parse_labels_header(ifstream &f);
void parse_images_data(ifstream &f, MNIST_metadata meta, vector<Mat> *mnist);
void parse_labels_data(ifstream &f, MNIST_metadata meta, vector<Label> *labels);
vector<Mat> load_images(string path);
vector<Label> load_labels(string path);

int main(int argc, char** argv)
{
    cout << "Creating train LMDB\n";
    create_lmdbs(TRAIN_IMAGES, TRAIN_LABELS, LMDB_TRAIN);
    cout << "Creating test LMDB\n";
    create_lmdbs(TEST_IMAGES, TEST_LABELS, LMDB_TEST);
    return 0;
}

void create_lmdbs(const char* images, const char* labels, const char* lmdb_path)
{
    /*LMDB related code was taken from Caffe script convert_mnist_data.cpp*/

    // lmdb data 
    MDB_env *mdb_env;
    MDB_dbi mdb_dbi;
    MDB_val mdb_key, mdb_data;
    MDB_txn *mdb_txn;

    // Set database environment
    mkdir(lmdb_path, 0744);

    // Create Data LMDB
    mdb_env_create(&mdb_env);
    mdb_env_set_mapsize(mdb_env, TB);
    mdb_env_open(mdb_env, lmdb_path, 0, 0664);
    mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn);
    mdb_open(mdb_txn, NULL, 0, &mdb_dbi);

    // Load images/labels
    vector<Mat> list_imgs = load_images(images);
    vector<Label> list_labels = load_labels(labels);

    vector< pair<Mat, Label> > imgs_labels(list_imgs.size());
    for (unsigned int i = 0; i<list_imgs.size(); i++)
    {
        imgs_labels[i] = pair<Mat, Label>(list_imgs[i], list_labels[i]);

    }
    random_shuffle(std::begin(imgs_labels), std::end(imgs_labels));

    // Dimensions of Data LMDB 
    unsigned int rows = list_imgs[0].rows;
    unsigned int cols = list_imgs[0].cols;

    int count = 0;
    string data_value, label_value;
    
    Datum datum;
    datum.set_channels(1);
    datum.set_height(rows);
    datum.set_width(cols);

    std::ostringstream s;

    for (unsigned int i = 0; i<LMDB_SIZE; i++)
    {
        s << std::setw(8) << std::setfill('0') << i; 
        string key_str = s.str();
        s.str(std::string());

        // Set Data
        datum.set_data((char*)list_imgs[i].data, rows*cols);
        datum.set_label((uint32_t)list_labels[i]);
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
        }
    }

    // Last batch
    if (count % 1000 != 0) {
        mdb_txn_commit(mdb_txn);
        mdb_close(mdb_env, mdb_dbi);
        mdb_env_close(mdb_env);
    }

    cout << "\nFinished creation of LMDB's\n";
    return;
}

vector<Mat> load_images(string path)
{
    ifstream f;
    f.open(path, ios::in | ios::binary);

    MNIST_metadata meta = parse_images_header(f);
    cout << "\nMagic number: " << meta.magic << endl; 
    cout << "Number of Images: " << meta.num_elems << endl; 
    cout << "Rows: " << meta.rows << endl;
    cout << "Columns: " << meta.cols << endl; 
    vector<Mat> mnist(meta.num_elems);
    parse_images_data(f, meta, &mnist);
    return mnist;
}

vector<Label> load_labels(string path)
{
    ifstream f;
    f.open(path, ios::in | ios::binary);

    MNIST_metadata meta = parse_labels_header(f);
    cout << "\nMagic number: " << meta.magic << endl; 
    cout << "Number of Labels: " << meta.num_elems << endl; 
    vector<Label> labels_mnist(meta.num_elems);
    parse_labels_data(f, meta, &labels_mnist);
    return labels_mnist;

}

MNIST_metadata parse_images_header(ifstream &f)
{
    MNIST_metadata meta;
    streampos offset = 0;
    meta.magic = get_uint32_t(f, offset);
    offset += sizeof(uint32_t);
    meta.num_elems = get_uint32_t(f, offset);
    offset += sizeof(uint32_t);
    meta.rows = get_uint32_t(f, offset);
    offset += sizeof(uint32_t);
    meta.cols = get_uint32_t(f, offset);
    return meta;
}

void parse_images_data(ifstream &f, MNIST_metadata meta, vector<Mat> *mnist)
{
    unsigned int size_img = meta.cols * meta.rows;
    // 4 integers in the header of the images file
    streampos offset = sizeof(uint32_t) * 4;
    for (unsigned int i=0; i<meta.num_elems; i++)
    {
        vector<uByte> raw_data = read_block(f, size_img, offset);
        Mat mchar(raw_data, false);
        mchar = mchar.reshape(1, meta.rows);
        (*mnist)[i] = mchar.clone();
        offset += size_img;
    }
}

MNIST_metadata parse_labels_header(ifstream &f)
{
    MNIST_metadata meta;
    streampos offset = 0;
    meta.magic = get_uint32_t(f, offset);
    offset += sizeof(uint32_t);
    meta.num_elems = get_uint32_t(f, offset);
    return meta;
}

void parse_labels_data(ifstream &f, MNIST_metadata meta, vector<Label> *labels)
{
    // 4 integers in the header of the images file
    streampos offset = sizeof(uint32_t) * 2;
    for (unsigned int i=0; i<meta.num_elems; i++)
    {
        f.seekg(offset);
        Label label;
        f.read((Byte*) &label, sizeof(Label));
        (*labels)[i] = label;
        offset += sizeof(Label);
    }
}

vector<uByte> read_block(ifstream &f, unsigned int size, streampos offset)
{
    Byte* bytes; 
    bytes = (Byte*) malloc(size*sizeof(uByte));

    f.seekg(offset);
    f.read(bytes, size);

    vector<uByte> raw_data(size);
    for (unsigned int i=0; i<size; i++)
    {
        raw_data[i] = (uByte) bytes[i];
    }

    free(bytes);

    return raw_data;
}
 
/*
 * It parses a int (32 bits) from the file f.
 * The MNIST dataset uses big-endian. This function take into account
 * wheter the local architecture is {big,little}-endian and return 
 * the correct interpretation of the integer.
 * 
 * Precondition: f has to be opened with ios::in | ios::binary flags
 */
uint32_t get_uint32_t(ifstream &f, streampos offset)
{
    // TODO add support to big-endian machines
    uint32_t* i_int;
    Byte* b_int; 

    b_int = (Byte*) malloc(sizeof(uint32_t));

    for (unsigned int i=0; i<sizeof(uint32_t); i++)
    {
        f.seekg(offset + (streampos) i);
        f.read(b_int+(sizeof(uint32_t)-i-1), sizeof(Byte));
    }
    i_int = reinterpret_cast<uint32_t*>(b_int); 

    uint32_t res = *i_int;
    free(b_int);

    return res;
}
