#include <iostream>
#include <string>
#include <sys/stat.h>
#include <cstdarg>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "caffe/proto/caffe.pb.h"

#define TB 1099511627776

using namespace std;
using namespace cv;

typedef char Byte;
typedef unsigned char uByte;
typedef uByte Label;

void CVMatsToDatum(const Mat &img1, const Mat &img2, Datum *datum);
void Mat2Datum(const Mat &img, Datum *datum);
void Mats2Datum(const Mat &img1, const Mat &img2, Datum *datum);

class LMDataBase {
public:
  /*************************************************************
   * Use cases:                                                *
   * LMDataBase(path, 3, 1)   for 3 int labels                 *
   * LMDataBase(path, 6, 224) for 3 channels images of 224x224 *
   * LMDataBase(path, 2, 28)  for 1 channel images of 28x28    *
   *************************************************************/
  LMDataBase(const char *lmdb_path, size_t dat_channels, size_t dat_size);
  ~LMDataBase() { close_env_lmdb(); };
  void insert2db(Mat &img);
  void insert2db(Mat &img1, Mat &img2);
  void insert2db(vector<Label> &labels);

private:
  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;
  size_t datum_size;
  size_t datum_channels;
  unsigned int num_inserts;

  void save_data_to_lmdb(string &data_value);
  void commit_data_to_lmdb();
  void close_env_lmdb(); 
