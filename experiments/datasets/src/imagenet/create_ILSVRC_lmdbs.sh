#!/usr/bin/env bash
im_per_class=("20" "1000")
files=("Testing" "Training")
if [ -z $1 ]
then
    echo "You must provide the root dir where the LMDBs are going to be stored"
    exit 1
fi

lmdb_root=$1
mkdir -p $lmdb_root
# Change paths if needed
for s in ${im_per_class[@]}; do 
    for t in ${files[@]}; do 
        ../utils/create_lmdb.sh -r 227\
            -l ${lmdb_root}/ILSVRC12_${t}_${s}perclass_lmdb\
            -d /\
            -f ./ILSVRC_${s}_${t}.txt
    done
done
