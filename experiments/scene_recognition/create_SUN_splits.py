#!/usr/bin/env python2.7
from sys import argv, exit
from os import makedirs
from os.path import isdir, join, dirname

DATA = "./data/paths"
SPLITS = [1, 2, 3]
SPLITS_NAMES = ["Training_", "Testing_"]

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
            splitfile = prefix + "0" * (split != 10) + str(split) + ".txt"
            with open(join(part_path, splitfile), 'r') as f:
                list = ["{} {}".format(elem, classes.index(dirname(elem)))
                        for elem in f.read().splitlines()]
                list = "\n".join(list)
                with open(join(DATA, splitfile), 'w') as out:
                    out.write(list)

    exit(0)
