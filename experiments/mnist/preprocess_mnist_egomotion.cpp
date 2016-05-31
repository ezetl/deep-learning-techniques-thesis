/*
 * This code parses the MNIST dataset files (images and labels).
 *
 * It was done for my little-endian machine, but you can set the LITTLE_ENDIAN
 * flag off and it will run in high endian mode (TODO)
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

#define L_ENDIAN true //LITTLE ENDIAN
#define TB 1099511627776
#define NUM_TRASLATIONS 7
#define NUM_ROTATIONS 61
#define NUM_BIN_ROTATIONS 20
#define NUM_CLASSES 3
#define LABEL_WIDTH NUM_BIN_ROTATIONS
#define LOWER_ANGLE -30
#define LOWER_TRASLATION -3
#define BATCHES 10 

#define DATA_ROOT    "../data/"
#define TRAIN_IMAGES (DATA_ROOT"train-images-idx3-ubyte")
#define TRAIN_LABELS (DATA_ROOT"train-labels-idx1-ubyte")

#define LMDB_ROOT         "../data/"
#define LMDB_TRAIN        (LMDB_ROOT"mnist_train_egomotion_lmdb/")

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

typedef struct
{
    Mat img;
    Label x;
    Label y;
    Label z;
} DataBlob;

void create_lmdbs(const char* images, const char* labels, const char* lmdb_path);
uint32_t get_uint32_t(ifstream &f, streampos offset);
vector<uByte> read_block(ifstream &f, unsigned int size, streampos offset);
MNIST_metadata parse_images_header(ifstream &f);
MNIST_metadata parse_labels_header(ifstream &f);
void parse_images_data(ifstream &f, MNIST_metadata meta, vector<Mat> *mnist);
void parse_labels_data(ifstream &f, MNIST_metadata meta, vector<Label> *labels);
Mat transform_image(Mat &img, float tx, float ty, float rot);
vector<Mat> load_images(string path);
vector<Label> load_labels(string path);
vector<DataBlob> process_images(vector<Mat> &list_imgs, unsigned int amount_pairs);
unsigned int generate_rand(int range_limit);

int main(int argc, char** argv)
{
    cout << "Creating train LMDB\n";
    create_lmdbs(TRAIN_IMAGES, TRAIN_LABELS, LMDB_TRAIN);
    return 0;
}

void create_lmdbs(const char* images, const char* labels, const char* lmdb_path)
{
    /* LMDB related code was adapted from Caffe script convert_mnist_data.cpp */
    /* We dont use labels in this case */

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
    random_shuffle(std::begin(list_imgs), std::end(list_imgs));


    // Dimensions of Data LMDB 
    unsigned int rows = list_imgs[0].rows;
    unsigned int cols = list_imgs[0].cols;

    int count = 0;
    string data_value, label_value;
    
    // Data datum
    // This datum has 2 + NUM_CLASSES dimensions.
    // 2 dimensiones because we are merging 2 one channel images into one, 
    // plus NUM_CLASSES (3) because we are also merging the labels into the data.
    // This allow us to slice the data later on training phase and retrieve the labels.
    // Basically I am adding 3 "images" with all zeros in it except the index where the class 
    // is active. Then with the Argmax Layer of Caffe I retrieve these labels index again and pass them 
    // to the loss function.
    // Chek the loop below to see how it is done.
    Datum datum;
    datum.set_channels(2+NUM_CLASSES);
    datum.set_height(rows);
    datum.set_width(cols);

    std::ostringstream s;

    // Processing and generating million of images at once will consume too much RAM (>7GB)
    // and it will (probably) throw a std::bad_alloc exception.
    // Lets split the processing in several batches instead. 
    // list_imgs.size() has to be multiple of BATCHES (to simplify things)
    int len_batch = list_imgs.size() / BATCHES;
    for (unsigned int i = 0; i<BATCHES; i++)
    {
        unsigned int begin = i * len_batch; 
        unsigned int end = begin + len_batch - 1;
        vector<Mat> batch_imgs = vector<Mat>(list_imgs.begin()+begin, list_imgs.begin()+end);
        unsigned int amount_pairs = 16;
        if (i==0 || i==1){
            amount_pairs = 17;
        } 
        vector<DataBlob> batch_data = process_images(batch_imgs, amount_pairs);
        for (unsigned int item_id = 0; item_id < batch_data.size(); ++item_id) {
            // Dont use item_id as key here since we have 83/85 images per original image,
            // meaning that we will overwrite the same image 83/85 times instead of creating 
            // a new entry
            s << std::setw(8) << std::setfill('0') << count; 
            string key_str = s.str();
            s.str(std::string());

            // Set Data
            // Create a char pointer and copy the images first, the labels at the end
            char * data_label;
            data_label = (char*)calloc(rows*cols*5, sizeof(uByte));
            data_label = (char*)memcpy(data_label, (void*)batch_data[item_id].img.data, 2*rows*cols);

            char * labels;
            unsigned int labelx = (unsigned int)batch_data[item_id].x;
            unsigned int labely = (unsigned int)batch_data[item_id].y;
            unsigned int labelz = (unsigned int)batch_data[item_id].z;
            labels = (char*) calloc(rows*cols*3, sizeof(uByte));
            labels[labelx] = 1;
            labels[cols*rows + labely] = 1;
            labels[cols*rows*2 + labelz] = 1;
            memcpy(data_label+(cols*rows*2), (void*)labels, 3*rows*cols);

            datum.set_data((char*)data_label, 5*rows*cols);
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
            if (count % 50000 == 0) {
                cout << "Processed " << count << "\r" << flush;
            }
            free(data_label);
            free(labels);
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

/*
 * rot (Rotation) is in degrees
 * tx, ty (Translations) are pixels
 */
Mat transform_image(Mat &img, float tx, float ty, float rot)
{
    Mat res;
    Point2f mid(img.cols / 2, img.rows / 2);
    Mat rotMat = getRotationMatrix2D(mid,  rot,  1.0);
    Mat transMat = (Mat_<double>(2,3) << 0,0,tx,0,0,ty);
    rotMat = rotMat + transMat;
    // Set constant value for border to be white
    warpAffine(img,
               res,
               rotMat,
               Size(img.cols, img.rows),
               INTER_LINEAR,
               BORDER_CONSTANT,
               Scalar(0, 0, 0));
    return res;
}

vector<DataBlob> process_images(vector<Mat> &list_imgs, unsigned int amount_pairs)
{
    vector<DataBlob> final_data;
    srand(0);
    unsigned int rand_index = 0;
    vector<float> translations(NUM_TRASLATIONS);
    float value = LOWER_TRASLATION;
    for (unsigned int i=0; i<translations.size(); i++)
    {
        translations[i] = value++;
    }

    value = LOWER_ANGLE;
    vector<float> rotations(NUM_ROTATIONS);
    for (unsigned int i=0; i<rotations.size(); i++)
    {
        rotations[i] = value++;
    }

    // Debugging
    //namedWindow("Normal");
    //namedWindow("Transformed");
    for (unsigned int i=0; i<list_imgs.size(); i++)
    {
        // Use white background
        //bitwise_not(list_imgs[i], list_imgs[i]);

        for (unsigned int j=0; j<amount_pairs; j++)
        {
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

            // Merge the original img and the transformed one into a unique Mat
            // Then we split the channels in Caffe using the SLICE layer
            auto channels = vector<Mat>{list_imgs[i], new_img};
            Mat merged_mats;
            merge(channels, merged_mats);
            d.img = merged_mats;

            final_data.push_back(d);

            // Debugging
            //imshow("Normal", list_imgs[i]);
            //imshow("Transformed", new_img);
            //waitKey(100);
        }
    }
    return final_data;
}

/* Generate a random number between 0 and range_limit-1
 * Useful to get a random element in an array of size range_limit
 */
unsigned int generate_rand(int range_limit)
{
    return rand() % range_limit;
}
