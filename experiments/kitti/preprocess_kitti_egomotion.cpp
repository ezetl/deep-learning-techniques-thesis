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
#define NUM_BINS 20
#define NUM_CLASSES 3
#define LABEL_WIDTH NUM_BINS
#define PAIRS_PER_SPLIT 500

#define DATA_ROOT    "../data/"
#define IMAGES       "/media/eze/0F4A13791A35DD40/KITTI/dataset/sequences/"

#define LMDB_ROOT       DATA_ROOT 
#define LMDB_TRAIN      (LMDB_ROOT"kitti_train_egomotion_lmdb/")
#define LMDB_VAL        (LMDB_ROOT"kitti_val_egomotion_lmdb/")

typedef char Byte;
typedef unsigned char uByte;
typedef uByte Label;
typedef struct
{
    Mat img1;
    Mat img2;
    Label x;
    Label y;
    Label z;
} DataBlob;

// 9 Sequences for training, 2 for validation
const vector<string> SPLITS = {"00.txt", "01.txt", "02.txt", "03.txt", "04.txt", "05.txt", "06.txt", "07.txt", "08.txt", "09.txt", "10.txt"};

unsigned int generate_rand(int range_limit);
void create_lmdbs(const char* images, const char* lmdb_train_path, const char* lmdb_val_path);
vector< vector< tuple<string, string> > > generate_pairs(const char* split);
vector<DataBlob> process_images(vector<Mat> &list_imgs, unsigned int amount_pairs);

vector< vector< tuple<string, string> > > generate_pairs(const vector<string> splits) {
    vector< vector< tuple<string, string> > > res;
    for (unsigned int i=0; i<splits.size(); ++i) {
        ifstream fsplit;
        fsplit.open(IMAGES+splits[i]);
        string path;
        // Load original paths
        vector<string> split_paths;
        while (fsplit >> path) {
            split_paths.push_back(IMAGES+path);
        }
        // Generate pairs
        vector< tuple<string, string> > pairs_paths;
        for (unsigned int j=0; j<PAIRS_PER_SPLIT; ++j) {
            int index = generate_rand(split_paths.size());
            int pair_index = 0;
            int pair_offset = generate_rand(7)+1;
            if (index==0) {
                pair_index = index + pair_offset;
            } else if (index == split_paths.size()-1) {
                pair_index = index - pair_offset;
            } else {
                if (generate_rand(2)) { // go to the left
                    // Careful with this substraction. If the 2 operands were unsigned int we could get in trouble (overflow)
                    pair_index = (index - pair_offset >= 0) ? index - pair_offset : 0;  
                } else { // go to the right
                    pair_index = (index + pair_offset <= split_paths.size()-1) ? index + pair_offset : split_paths.size()-1;  
                }
            }
            tuple<string, string> pair = make_tuple(split_paths[index], split_paths[pair_index]);
            pairs_paths.push_back(pair);
        }
        res.push_back(pairs_paths);
    }
    return res;
}

int main(int argc, char** argv)
{
    cout << "Creating train/val LMDB's\n";
    create_lmdbs(IMAGES, LMDB_TRAIN, LMDB_VAL);
    return 0;
}

void create_lmdbs(const char* images, const char* lmdb_train_path, const char* lmdb_val_path)
{
    // lmdb data ('v' suffix is for validation LMDB) 
    MDB_env *mdb_env, *mdb_envv;
    MDB_dbi mdb_dbi, mdb_dbiv;
    MDB_val mdb_key, mdb_data, mdb_keyv, mdb_datav;
    MDB_txn *mdb_txn, *mdb_txnv;

    // Set database environment
    mkdir(lmdb_train_path, 0744);
    mkdir(lmdb_val_path, 0744);

    // Create Data LMDBs
    mdb_env_create(&mdb_env);
    mdb_env_set_mapsize(mdb_env, TB);
    mdb_env_open(mdb_env, lmdb_train_path, 0, 0664);
    mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn);
    mdb_open(mdb_txn, NULL, 0, &mdb_dbi);

    mdb_env_create(&mdb_envv);
    mdb_env_set_mapsize(mdb_envv, TB);
    mdb_env_open(mdb_envv, lmdb_val_path, 0, 0664);
    mdb_txn_begin(mdb_envv, NULL, 0, &mdb_txnv);
    mdb_open(mdb_txnv, NULL, 0, &mdb_dbiv);

    // Generate pairs of images for each sequence 
    vector< vector< tuple<string, string> > > train_pairs = generate_pairs(SPLITS);
    vector< vector< tuple<string, string> > >::iterator it;
    vector< tuple<string, string> >::iterator it2;

    for(it=train_pairs.begin(); it!=train_pairs.end(); ++it){
        for (it2=(*it).begin(); it2!=(*it).end(); ++it2){
            cout << get<0>(*it2) << " " << get<1>(*it2) << endl;
        }
    } 
    //random_shuffle(std::begin(list_imgs), std::end(list_imgs));


    /*
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
    */
    return;
}

vector<DataBlob> process_images(vector<Mat> &list_imgs, unsigned int amount_pairs)
{
    vector<DataBlob> final_data;
    /*
    srand(0);
    unsigned int rand_index = 0;
    vector<float> translations(7);
    float value = 0;
    for (unsigned int i=0; i<translations.size(); i++)
    {
        translations[i] = value++;
    }

    value = 100;
    vector<float> rotations(100);
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
            rand_index = generate_rand(100);
            d.x = rand_index; 
            float tx = translations[rand_index];
            // Generate random Y translation
            rand_index = generate_rand(100);
            d.y = rand_index; 
            float ty = translations[rand_index];
            // Calculate random bin of rotation (0 to 19)
            rand_index = generate_rand(1000);
            d.z = rand_index;
            // Calculate the real index of the array of rotations (0 to 61)
            rand_index *= 3;
            rand_index += generate_rand(3);
            float rot = rotations[rand_index];

            // Finally, apply the selected transformations to the image
            //Mat new_img = transform_image(list_imgs[i], tx, ty, rot);

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
