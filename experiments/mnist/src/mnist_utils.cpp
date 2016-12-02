#include "mnist_utils.hpp"

typedef struct {
  uint32_t magic;
  uint32_t num_elems;
  uint32_t cols;
  uint32_t rows;
} MNIST_metadata;

uint32_t get_uint32_t(ifstream &f, streampos offset);
vector<unsigned char> read_block(ifstream &f, unsigned int size, streampos offset);
MNIST_metadata parse_images_header(ifstream &f);
MNIST_metadata parse_labels_header(ifstream &f);
void parse_images_data(ifstream &f, MNIST_metadata meta, vector<Mat> *mnist);
void parse_labels_data(ifstream &f, MNIST_metadata meta, vector<unsigned char> *labels);

vector<Mat> load_images(string path) {
  ifstream f;
  f.open(path, ios::in | ios::binary);

  MNIST_metadata meta = parse_images_header(f);
  cout << "MNIST data info:" << endl;
  cout << "  Magic number: " << meta.magic << endl;
  cout << "  Number of Images: " << meta.num_elems << endl;
  cout << "  Rows: " << meta.rows << endl;
  cout << "  Columns: " << meta.cols << endl;
  vector<Mat> mnist(meta.num_elems);
  parse_images_data(f, meta, &mnist);
  return mnist;
}

vector<unsigned char> load_labels(string path) {
  ifstream f;
  f.open(path, ios::in | ios::binary);

  MNIST_metadata meta = parse_labels_header(f);
  cout << "MNIST labels info:" << endl;
  cout << "  Magic number: " << meta.magic << endl;
  cout << "  Number of Labels: " << meta.num_elems << endl;
  vector<unsigned char> labels_mnist(meta.num_elems);
  parse_labels_data(f, meta, &labels_mnist);
  return labels_mnist;
}

MNIST_metadata parse_images_header(ifstream &f) {
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

void parse_images_data(ifstream &f, MNIST_metadata meta, vector<Mat> *mnist) {
  unsigned int size_img = meta.cols * meta.rows;
  // 4 integers in the header of the images file
  streampos offset = sizeof(uint32_t) * 4;
  for (unsigned int i = 0; i < meta.num_elems; i++) {
    vector<unsigned char> raw_data = read_block(f, size_img, offset);
    Mat mchar(raw_data, false);
    mchar = mchar.reshape(1, meta.rows);
    (*mnist)[i] = mchar.clone();
    offset += size_img;
  }
}

MNIST_metadata parse_labels_header(ifstream &f) {
  MNIST_metadata meta;
  streampos offset = 0;
  meta.magic = get_uint32_t(f, offset);
  offset += sizeof(uint32_t);
  meta.num_elems = get_uint32_t(f, offset);
  return meta;
}

void parse_labels_data(ifstream &f, MNIST_metadata meta, vector<unsigned char> *labels) {
  // 4 integers in the header of the images file
  streampos offset = sizeof(uint32_t) * 2;
  for (unsigned int i = 0; i < meta.num_elems; i++) {
    f.seekg(offset);
    unsigned char label;
    f.read((char *)&label, sizeof(unsigned char));
    (*labels)[i] = label;
    offset += sizeof(unsigned char);
  }
}

vector<unsigned char> read_block(ifstream &f, unsigned int size, streampos offset) {
  char *bytes;
  bytes = (char *)malloc(size * sizeof(unsigned char));

  f.seekg(offset);
  f.read(bytes, size);

  vector<unsigned char> raw_data(size);
  for (unsigned int i = 0; i < size; i++) {
    raw_data[i] = (unsigned char)bytes[i];
  }

  free(bytes);

  return raw_data;
}

/*
 * It parses a int (32 bits) from the file f.
 * The MNIST dataset uses big-endian. It parses assuming the
 * host is little-endian
 *
 * PRECONDITION: f has to be opened with ios::in | ios::binary flags
 */
uint32_t get_uint32_t(ifstream &f, streampos offset) {
  uint32_t *i_int;
  char *b_int;

  b_int = (char *)malloc(sizeof(uint32_t));

  for (unsigned int i = 0; i < sizeof(uint32_t); i++) {
    f.seekg(offset + (streampos)i);
    f.read(b_int + (sizeof(uint32_t) - i - 1), sizeof(char));
  }
  i_int = reinterpret_cast<uint32_t *>(b_int);

  uint32_t res = *i_int;
  free(b_int);

  return res;
}
