#!/usr/bin/env python2.7
from sys import argv, exit
from os import makedirs
from os.path import isdir, join, dirname
from collections import defaultdict

DATA = "./data/paths"
SPLITS = [1, 2, 3]
SPLITS_NAMES = ["Training_", "Testing_"]
IMGS_PER_CLASS = [5, 20]

if __name__ == "__main__":
    if len(argv) < 2:
        exit("You have to provide the path to the Partitions folder.\n\
             \rYou can download SUN dataset and the Partitions folder\n\
             \rfrom the official website:\n\
             \r\nhttp://vision.princeton.edu/projects/2010/SUN/\n\n")

    if not isdir(DATA):
        makedirs(DATA)

    part_path = argv[1]
    if not isdir(part_path):
        exit("Provided path is not a folder.\n")

    with open(join(part_path, "ClassName.txt"), 'r') as f:
        classes = f.read().splitlines()

    for split in SPLITS:
        for prefix in SPLITS_NAMES:
            for im_per_class in IMGS_PER_CLASS:
                splitfile = prefix + "0" * (split != 10) + str(split)
                outfile = splitfile + "_" + str(im_per_class) + "per_class.txt"
                with open(join(part_path, splitfile+".txt"), 'r') as f:
                    classes_dict = defaultdict(list)
                    for elem in f.read().splitlines():
                        classes_dict[classes.index(dirname(elem))].append(elem)
                    lists_of_imgs = [zip(classes_dict[key][0:im_per_class], [key]*im_per_class) for key in classes_dict]
                    flattened = ["{} {}".format(item[0], item[1]) for sublist in lists_of_imgs for item in sublist]
                    flattened = "\n".join(flattened)
                    with open(join(DATA, outfile), 'w') as out:
                        out.write(flattened)
    exit(0)
