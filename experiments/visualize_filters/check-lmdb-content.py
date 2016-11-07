#!/usr/bin/env python2.7
"""
If you try to run this script an receive an error like:

Gtk-ERROR **: GTK+ 2.x symbols detected. Using GTK+ 2.x and GTK+ 3 in the same process is not supported

Then uncomment the import gtk line
"""
import gtk
import caffe
import lmdb
import numpy as np
import sys
import cv2

np.set_printoptions(threshold='nan', linewidth=200)


def print_lmdb_data(lmdb_path):
    lmdb_env = lmdb.open(lmdb_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()

    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        data = caffe.io.datum_to_array(datum)
        dims = data.shape[0]
        # If number of dims is 5, it is clearly mnist, since we have 2 images
        # of one channel each and 3 more channels for the labels
        # If number of dims is 6, it is a data lmdb with 2 images of 3 channels
        # each. This is the new method I'm using to create databases: one LMDB
        # for data, othe LMDB for labels
        if dims == 5:
            cv2.imshow('im1', data[0])
            cv2.imshow('im2', data[1])
            cv2.waitKey(0)
        elif dims == 6:
            im = data.astype(np.uint8)
            im1 = im[0:3]
            im2 = im[3:6]
            im1 = np.transpose(im1, (1, 2, 0))
            im2 = np.transpose(im2, (1, 2, 0))
            cv2.imshow('im1', im1)
            cv2.imshow('im2', im2)
            cv2.waitKey(0)
        elif dims == 3:
            # Assume its a standar lmdb with a 3 channel image
            im = data.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            cv2.imshow('im1', im)
            cv2.waitKey(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("You have to provide the path to the lmdb\n")

    lmdb_path = sys.argv[1]
    print_lmdb_data(lmdb_path)
