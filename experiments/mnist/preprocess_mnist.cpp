#include <iostream>
#include <fstream>
#include <cinttypes>

/*
 * This code parses the MNIST dataset files (images and labels).
 * It was done for my low-endian machine, but you can set the LOW_ENDIAN
 * flag off and it will run in high endian mode
 *
 * Compile:
 *     g++ -o preprocess_mnist preprocess_mnist.cpp -std=gnu++11
 */

using namespace std;

#define LOW_ENDIAN true

#define DATA_ROOT    "data/"
#define TRAIN_IMAGES (DATA_ROOT"train-images-idx3-ubyte")
#define TRAIN_LABELS (DATA_ROOT"train-labels-idx1-ubyte")
#define TEST_IMAGES  (DATA_ROOT"t10k-images-idx3-ubyte")
#define TEST_LABELS  (DATA_ROOT"t10k-labels-idx1-ubyte")

typedef char Byte;
typedef struct
{
int32_t magic;
int32_t num_elems;
int32_t cols;
int32_t rows;
} MNIST_metadata;

int32_t get_int32_t(ifstream &f, streampos offset);
MNIST_metadata parse_images_header(ifstream &f);
MNIST_metadata parse_labels_header(ifstream &f);
void load_images(string path);
void load_labels(string path);
void process_images();
void process_labels();

int main()
{
    load_images(TRAIN_IMAGES);
    
    return 0;
}

void load_images(string path)
{
    ifstream f;
    f.open(path, ios::in | ios::binary);

    MNIST_metadata meta = parse_images_header(f);
    cout << "Magic number: " << meta.magic << endl; 
    cout << "Number of Images: " << meta.num_elems << endl; 
    cout << "Rows: " << meta.rows << endl;
    cout << "Columns: " << meta.cols << endl; 
}

MNIST_metadata parse_labels_header(ifstream &f)
{
    MNIST_metadata meta;
    streampos offset = 0;
    meta.magic = get_int32_t(f, offset);
    offset += sizeof(int32_t);
    meta.num_elems = get_int32_t(f, offset);
    return meta;
}

MNIST_metadata parse_images_header(ifstream &f)
{
    MNIST_metadata meta;
    streampos offset = 0;
    meta.magic = get_int32_t(f, offset);
    offset += sizeof(int32_t);
    meta.num_elems = get_int32_t(f, offset);
    offset += sizeof(int32_t);
    meta.rows = get_int32_t(f, offset);
    offset += sizeof(int32_t);
    meta.cols = get_int32_t(f, offset);
    return meta;
} 

/*
 * It parses a int (32 bits) from the file f.
 * The MNIST dataset uses big-endian. This function take into account
 * wheter the local architecture is {big,little}-endian and return 
 * the correct interpretation of the integer.
 * 
 * Precondition: f has to be opened with ios::in | ios::binary flags
 */
int32_t get_int32_t(ifstream &f, streampos offset)
{
    int32_t* i_int;
    Byte* b_int; 

    b_int = (Byte*) malloc(sizeof(int32_t));

    for (unsigned int i=0; i<sizeof(int32_t); i++)
    {
        f.seekg(offset + (streampos) i);
        f.read(b_int+(sizeof(int32_t)-i-1), sizeof(Byte));
    }
    i_int = reinterpret_cast<int32_t*>(b_int); 

    int32_t res = *i_int;
    free(b_int);

    return res;
}

void load_labels(string path)
{

}

void process_images()
{

}

void process_labels()
{

}
