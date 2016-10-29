#!/usr/bin/env python2.7
import caffe
import lmdb
import numpy as np
import sys

np.set_printoptions(threshold='nan', linewidth=200)


def print_mnist_lmdb(lmdb_path):
    lmdb_env = lmdb.open(lmdb_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()

    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        data = caffe.io.datum_to_array(datum)
        dims = data.shape[0]
        for d in range(dims):
            print(data[d])
            print("\n\n\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("You have to provide the path to the MNIST lmdb\n\
              \ryou want to print on console.")
        sys.exit(1)

    lmdb_path = sys.argv[1]
    try:
        print_mnist_lmdb(lmdb_path)
    except KeyboardInterrupt:
        print("\n")
        sys.exit(1)
