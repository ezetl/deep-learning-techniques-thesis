#include <iostream>
#include <string>
#include <sys/stat.h>
#include <cstdarg>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include "caffe/proto/caffe.pb.h"

#define TB 1099511627776

using namespace std;

class LMDataBase {
public:
  LMDataBase(const char *lmdb_path, size_t dat_channels, size_t dat_size);
  bool insert2db(...);

private:
  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;
  Datum datum;
  size_t datum_size;
  size_t datum_channels;
};
