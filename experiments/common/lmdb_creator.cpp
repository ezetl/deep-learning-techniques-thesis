#include "lmdb_creator.hpp"

LMDataBase::LMDataBase(const char *lmdb_path, size_t dat_channels, size_t dat_size) : datum_channels(num_channels), datum_size(dsize) {
  // Set database environment
  mkdir(lmdb_path, 0744);
  // Create LMDB
  mdb_env_create(&mdb_env);
  mdb_env_set_mapsize(mdb_env, TB);
  mdb_env_open(mdb_env, lmdb_path, 0, 0664);
  mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn);
  mdb_open(mdb_txn, NULL, 0, &mdb_dbi);

  datum.set_channels(datum_channels);
  datum.set_height(datum_size);
  datum.set_width(datum__size);
}

bool LMDataBase::insert2db(...) { return false; }
