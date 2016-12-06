#ifndef __LMDB_CREATOR__
#define __LMDB_CREATOR__
#include <iostream>
#include <string>
#include <iomanip>
#include <sys/stat.h>
#include <cstdarg>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#define TB 1099511627776

using namespace std;
using namespace cv;
using namespace caffe;

typedef char Byte;
typedef unsigned char Label;

void Mats2Datum(const Mat &img1, const Mat &img2, Datum *datum);
void Mat2Datum(const Mat &img, Datum *datum);

class LMDataBase {
public:
  /*************************************************************
   * This class helps you create a LMDB database for Caffe     *
   * While a LMDB can be easily built by using Caffe's default *
   * tools (<Caffe_path>/tools/convert_imageset.cpp), it       *
   * becomes difficult to deal with pairs of images for        *
   * siamese networks, specially when those pairs have to be   *
   * preprocessed and saved in order.                          *
   * Using the default tool I would have needed to create a    *
   * single LMDB for each two input layers of the siamese      *
   * networks, and some extras LMDBs for each label. By using  *
   * this library, the preprocessing of the images and the     *
   * creation of the LMDBs can be done in a single C++ script. *
   * Also, this approach reduces the amount of lmdbs to two:   *
   * one for the images, and one for the labels.               *
   *                                                           *
   * This class is also helpful to create regular LMDBS for    *
   * standar CNNs (see method insert2db(Mat &img)).            *
   *                                                           *
   * Please do read the methods implementations to see how I   *
   * merge two images in one and save them in the database     *
   * for later processing with CNNs                            *
   *                                                           *
   * Use cases:                                                *
   * LMDataBase(path, 3, 1)   for 3 int labels                 *
   * LMDataBase(path, 6, 224) for 3 channels images of 224x224 *
   * LMDataBase(path, 2, 28)  for 1 channel images of 28x28    *
   *************************************************************/
  LMDataBase(string lmdb_path, size_t dat_channels, size_t dat_size);
  ~LMDataBase() {
    close_env_lmdb();
    cout << "\nFinished creation of LMDB with " << num_inserts << " pairs of images.\n";
  };
  void insert2db(const Mat &img, int label);
  void insert2db(const Mat &img1, const Mat &img2, int label);
  void insert2db(const vector<Label> &labels);

private:
  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;
  size_t datum_channels;
  size_t datum_size;
  unsigned int num_inserts;

  void save_data_to_lmdb(string &data_value);
  void commit_data_to_lmdb();
  void close_env_lmdb(); 
};
#endif
