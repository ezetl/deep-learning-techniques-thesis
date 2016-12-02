#include "lmdb_creator.hpp"

LMDataBase::LMDataBase(string lmdb_path, size_t dat_channels, size_t dat_size)
    : datum_channels(dat_channels), datum_size(dat_size), num_inserts(0) {
  // Set database environment
  mkdir(static_cast<const char *>(lmdb_path.c_str()), 0744);
  // Create LMDB
  mdb_env_create(&mdb_env);
  mdb_env_set_mapsize(mdb_env, TB);
  mdb_env_open(mdb_env, static_cast<const char *>(lmdb_path.c_str()), 0, 0664);
  mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn);
  mdb_open(mdb_txn, NULL, 0, &mdb_dbi);
}

void LMDataBase::insert2db(Mat &img, int label = -10) {
  assert((size_t)img.cols == datum_size);
  assert((size_t)img.rows == datum_size);
  assert((size_t)img.channels() == datum_channels);

  string data_value;
  Datum datum;
  Mat2Datum(img, &datum);
  if (label != -10) {
    datum.set_label(label);
  }
  datum.SerializeToString(&data_value);

  save_data_to_lmdb(data_value);
  cout << "Processed " << ++num_inserts << "\r" << flush;
}

void LMDataBase::insert2db(Mat &img1, Mat &img2, int label = -10) {
  assert((size_t)img1.cols == datum_size);
  assert((size_t)img1.rows == datum_size);
  assert((size_t)img2.cols == datum_size);
  assert((size_t)img2.rows == datum_size);
  assert((size_t)img1.channels() == datum_channels);
  assert((size_t)img2.channels() == datum_channels);

  string data_value;
  Datum datum;
  Mats2Datum(img1, img2, &datum);
  if (label != -10) {
    datum.set_label(label);
  }
  datum.SerializeToString(&data_value);

  save_data_to_lmdb(data_value);
  cout << "Processed " << ++num_inserts << "\r" << flush;
}

void LMDataBase::insert2db(vector<Label> &labels) {
  assert(labels.size() == datum_channels);

  string label_value;
  Datum datum;
  datum.set_data(reinterpret_cast<char*>(&labels[0]), datum_channels);
  datum.SerializeToString(&label_value);

  save_data_to_lmdb(label_value);

  ++num_inserts;
}

void LMDataBase::save_data_to_lmdb(string &data_value) {
  // Get primary key for database
  std::ostringstream s;
  s << std::setw(8) << std::setfill('0') << num_inserts;
  string key = s.str();

  mdb_data.mv_size = data_value.size();
  mdb_data.mv_data = reinterpret_cast<void *>(&data_value[0]);
  mdb_key.mv_size = key.size();
  mdb_key.mv_data = reinterpret_cast<void *>(&key[0]);
  mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0);
  if (num_inserts % 1000 == 0) {
    commit_data_to_lmdb();
  }
}

void LMDataBase::commit_data_to_lmdb() {
  mdb_txn_commit(mdb_txn);
  mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn);
}

void LMDataBase::close_env_lmdb(){
  mdb_txn_commit(mdb_txn);
  mdb_close(mdb_env, mdb_dbi);
  mdb_env_close(mdb_env);
}

void Mats2Datum(const Mat &img1, const Mat &img2, Datum *datum) {
  // Modified from CVMatToDatum from Caffe
  // TODO: merge these two methods somehow
  assert(img1.depth() == CV_8U);
  assert(img2.depth() == CV_8U);
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
    const uchar *ptr1 = img1.ptr<uchar>(h);
    const uchar *ptr2 = img2.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels / 2; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr1[img_index]);
        datum_index = ((c + datum_channels / 2) * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr2[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

void Mat2Datum(const Mat &img, Datum *datum) {
  assert(img.depth() == CV_8U);
  datum->set_channels(img.channels());
  datum->set_height(img.rows);
  datum->set_width(img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  string buffer(datum_size, ' ');

  for (int h = 0; h < datum_height; ++h) {
    const uchar *ptr = img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index]);
      }
    }
  }
  datum->set_data(buffer);
}
