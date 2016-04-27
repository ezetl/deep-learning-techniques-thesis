#include <iostream>
#include <fstream>
#include <vector>
#include <cinttypes>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <string> 
/*
 * Author: Ezequiel Torti Lopez
 *
 * This code parses the MNIST dataset files (images and labels).
 * It was done for my low-endian machine, but you can set the LOW_ENDIAN
 * flag off and it will run in high endian mode
 *
 * How to compile:
 *     g++ -o preprocess_mnist preprocess_mnist.cpp -std=gnu++11 -lopencv_core -lopencv_highgui 
 *
 */

using namespace std;
using namespace cv;

#define LOW_ENDIAN true

#define DATA_ROOT    "../data/"
#define TRAIN_IMAGES (DATA_ROOT"train-images-idx3-ubyte")
#define TRAIN_LABELS (DATA_ROOT"train-labels-idx1-ubyte")
#define TEST_IMAGES  (DATA_ROOT"t10k-images-idx3-ubyte")
#define TEST_LABELS  (DATA_ROOT"t10k-labels-idx1-ubyte")

typedef char Byte;
typedef unsigned char uByte;
typedef struct
{
    uint32_t magic;
    uint32_t num_elems;
    uint32_t cols;
    uint32_t rows;
} MNIST_metadata;

uint32_t get_uint32_t(ifstream &f, streampos offset);
vector<uByte> read_block(ifstream &f, unsigned int size, streampos offset);
MNIST_metadata parse_images_header(ifstream &f);
MNIST_metadata parse_labels_header(ifstream &f);
void parse_images_data(ifstream &f, MNIST_metadata meta, vector<Mat> &mnist);
void parse_labels_data(ifstream &f, MNIST_metadata meta, vector<uByte> &labels);
vector<Mat> load_images(string path);
vector<uByte> load_labels(string path);
void process_images();
void process_labels();

int main()
{
    cout << "Loading training split: \n"; 
    vector<Mat> train_imgs = load_images(TRAIN_IMAGES);
    cout << "Loading testing split: \n"; 
    vector<Mat> test_imgs = load_images(TEST_IMAGES);
    cout << "Loading training labels: \n"; 
    vector<uByte> train_labels = load_labels(TRAIN_LABELS);
    cout << "Loading testing labels: \n"; 
    vector<uByte> test_labels = load_labels(TEST_LABELS);
    // TODO: add random rotation/translations.
    // TODO: save Mat to lmdb.
    return 0;
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
    parse_images_data(f, meta, mnist);
    return mnist;
}

vector<uByte> load_labels(string path)
{
    ifstream f;
    f.open(path, ios::in | ios::binary);

    MNIST_metadata meta = parse_labels_header(f);
    cout << "\nMagic number: " << meta.magic << endl; 
    cout << "Number of Labels: " << meta.num_elems << endl; 
    vector<uByte> labels_mnist(meta.num_elems);
    parse_labels_data(f, meta, labels_mnist);
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

void parse_images_data(ifstream &f, MNIST_metadata meta, vector<Mat> &mnist)
{
    unsigned int size_img = meta.cols * meta.rows;
    // 4 integers in the header of the images file
    streampos offset = sizeof(uint32_t) * 4;
    for (unsigned int i=0; i<meta.num_elems; i++)
    {
        vector<uByte> raw_data = read_block(f, size_img, offset);
        Mat mchar(raw_data, false);
        mchar = mchar.reshape(1, meta.rows);
        mnist[i] = mchar;
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

void parse_labels_data(ifstream &f, MNIST_metadata meta, vector<uByte> &labels)
{
    // 4 integers in the header of the images file
    streampos offset = sizeof(uint32_t) * 2;
    for (unsigned int i=0; i<meta.num_elems; i++)
    {
        f.seekg(offset);
        uByte label;
        f.read((Byte*) &label, sizeof(uByte));
        labels[i] = label;
        offset += sizeof(uByte);
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

void process_images()
{

}

void process_labels()
{

}
