#include "lmdb_creator.hpp"

LMDataBase::LMDataBase(const char *lmdb_path, size_t dat_channels,
                       size_t dat_size)
    : datum_channels(num_channels), datum_size(dsize), num_inserts(0) {
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

void LMDataBase::insert2db(Mat &img) {
  assert((size_t)img.cols == datum_size);
  assert((size_t)img.rows == datum_size);
  assert((size_t)img.channels() == datum_channels);

  string data_value;
  Datum datum;
  Mat2Datum(img, &datum);
  datum.SerializeToString(&data_value);

  save_data_to_lmdb(data_value);
  cout << "Processed " << count << "\r" << flush;

  num_inserts++;
}

void LMDataBase::insert2db(Mat &img1, Mat &img2) {
  assert((size_t)img1.cols == datum_size);
  assert((size_t)img1.rows == datum_size);
  assert((size_t)img2.cols == datum_size);
  assert((size_t)img2.rows == datum_size);
  assert((size_t)img1.channels() == datum_channels);
  assert((size_t)img2.channels() == datum_channels);

  string data_value;
  Datum datum;
  Mats2Datum(img1, img2, &datum);
  datum.SerializeToString(&data_value);

  save_data_to_lmdb(data_value);
  cout << "Processed " << count << "\r" << flush;

  num_inserts++;
}

void LMDataBase::insert2db(vector<Label> &labels) {
  assert(labels.size() == datum_channels);

  string label_value;
  Datum datum;
  datum.set_data(reinterpret_cast<char*>(&labels[0]), datum_channels);
  datum.SerializeToString(&label_value);

  save_data_to_lmdb(label_value);

  num_inserts++;
}

void LMDataBase::save_data_to_lmdb(string &data_value) {
  // Get primary key for database
  std::ostringstream s;
  s << std::setw(8) << std::setfill('0') << num_inserts;
  string key = s.str();

  mdb_data.mv_size = data_value.size();
  mdb_data.mv_data = reinterpret_cast<void *>(&data_value[0]);
  mdb_key.mv_size = key_str.size();
  mdb_key.mv_data = reinterpret_cast<void *>(&key_str[0]);
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
  CHECK(img1.depth() == CV_8U) << "Image data type must be unsigned byte";
  CHECK(img2.depth() == CV_8U) << "Image data type must be unsigned byte";
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
  CHECK(img.depth() == CV_8U) << "Image data type must be unsigned byte";
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
