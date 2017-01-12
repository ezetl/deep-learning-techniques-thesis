#!/usr/bin/env python2.7
from os import walk
from os.path import join
from random import seed, sample, shuffle
from sys import argv, exit


def get_train_test_splits(path, num_imgs_per_class):
    train = []
    test = []
    test_imgs_per_class = 2
    class_ind = 0
    for root, dirs, files in walk(path):
        if files:
            # generate a random sample of num_imgs_per_class
            # add extra images for testing
            num_imgs = min(num_imgs_per_class+test_imgs_per_class, len(files))
            sample_imgs = sample(files, num_imgs)
            sample_imgs = map(lambda x: (join(root, x), class_ind), sample_imgs)
            train += sample_imgs[:-test_imgs_per_class]
            test += sample_imgs[-test_imgs_per_class:]
            class_ind += 1
    shuffle(train)        
    shuffle(test)
    return train, test


if __name__ == "__main__":
    if len(argv) < 2:
        exit("You have to provide the root dir where the ILSVRC'12\n\
                \rdataset is stored. The ILSVRC folder should contain 1000 subfolders\n\
                \rlike 'n02091244', 'n02091467', etc. Example:\n\n\
                \r{} path/to/ILSVRC12".format(argv[0]))
    # Make the generation of the dataset reproducible
    seed(123)
    ilsvrc_path = argv[1]
    for num_imgs_per_class in [20, 1000]:
        train, test = get_train_test_splits(ilsvrc_path, num_imgs_per_class)
        with open("ILSVRC_{}_Training.txt".format(num_imgs_per_class), 'w') as f:
            f.write('\n'.join('{} {}'.format(*t) for t in train))
        with open("ILSVRC_{}_Testing.txt".format(num_imgs_per_class), 'w') as f:
            f.write('\n'.join('{} {}'.format(*t) for t in test))

