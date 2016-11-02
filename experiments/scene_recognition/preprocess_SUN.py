#!/usr/bin/env python2.7
from sys import exit, argv
from os import makedirs
from os.path import isdir, isfile, join, dirname
from multiprocessing import Pool
import cv2


def preprocess_split(split):
    """
    Takes a list of paths of (original_image, dst_image) and crops them by the
    center using the smallest dimension of (x,y).
    """
    for orig, dst in split:
        imorig = cv2.imread(orig)
        if imorig is None:
            print("Couldn't load image {}".format(orig))
            continue
        rows, cols, _ = imorig.shape
        start = abs(rows - cols) / 2
        if rows > cols:
            imorig = imorig[start:start+cols, 0:cols]
        else:
            imorig = imorig[0:rows, start:start+rows]
        cv2.imwrite(dst, imorig)


if __name__ == "__main__":
    if len(argv) < 2:
        exit("You have to provide the file with the list of images\n\
             \rand the /path/to/images/root:\n\
             \r{} files_list.txt /path/where/images/are/stored.\n".
             format(argv[0]))

    file_paths = argv[1]
    images_root = argv[2]

    if not isfile(file_paths):
        exit("List of images provided is not an existing file.")

    if not isdir(images_root):
        exit("Images dir is not an existing directory.")

    dst_root = join(images_root, "preprocessed")
    if not isdir(dst_root):
        makedirs(dst_root)

    with open(file_paths, 'r') as f:
        list = []
        for line in f.read().splitlines():
            path = line.split()[0][1:]
            orig = join(images_root, path)
            dst = join(dst_root, path)
            if not isdir(dirname(dst)):
                makedirs(dirname(dst))
            list.append((orig, dst))

    # Split the list in 8 chunks, each one will be processed by a CPU core
    nchunks = 8
    schunk = (len(list) + nchunks) / nchunks
    splits = [list[i:i + schunk] for i in xrange(0, len(list), schunk)]
    pool = Pool(nchunks)
    pool.map(preprocess_split, splits)
    pool.close()
    print("Finished processing.")
    exit(0)
