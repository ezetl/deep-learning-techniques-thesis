#!/usr/bin/env bash
im_per_class=("5" "20")
files=("Testing" "Training")

data_root=$1
if [ -z $data_root ]
then
    echo "You have to provide the path to the root folder of your SUN397 dataset"
    echo "(where your Partitions/ originals/ and preprocessed/ folders are)"
    echo "Example:"
    echo "$0 /media/eze/Datasets/SUN397"
    exit 0
fi

# Create lmdb root dir
mkdir -p ${data_root}/lmdbs

# Get path of this script, so we can locate the 'utils' and 'data' folders
local_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Iterate over all the splits we want to use and create Training/Testing lmdbs for each one
for i in $(seq 1 3); do
    for s in ${im_per_class[@]}; do 
        for t in ${files[@]}; do 
            $local_dir/utils/create_lmdb.sh -r 227\
                -l ${data_root}/lmdbs/SUN_${t}_0${i}_${s}perclass_lmdb\
                -d ${data_root}/preprocessed\
                -f $local_dir/data/paths/${t}_0${i}_${s}per_class.txt
        done
    done
done
