#!/usr/bin/env bash
im_per_class=("5" "20")
files=("Testing" "Training")
# data_root is where you downloaded SUN397
# data_root/preprocessed is where the cropped images live
# Change paths if needed
data_root=/media/eze/0F4A13791A35DD40/SUN397/

for i in $(seq 1 3); do
    for s in ${im_per_class[@]}; do 
        for t in ${files[@]}; do 
            ./utils/create_lmdb.sh -r 227\
                -l ${data_root}lmdbs/SUN_${t}_0${i}_${s}perclass_lmdb\
                -d ${data_root}preprocessed\
                -f data/paths/${t}_0${i}_${s}per_class.txt
        done
    done
done
