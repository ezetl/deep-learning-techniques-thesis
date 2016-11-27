#!/usr/bin/env bash 

# Creates a LMDB
# Author: Ezequiel Torti Lopez

# Getting options
while getopts ":r:l:d:f:" opt;
do
    case $opt in
        r) resize="$OPTARG"
        ;;
        l) lmdb_path="$OPTARG"
        ;;
        d) data_root="$OPTARG"
        ;;
        f) imgs_list="$OPTARG"
        ;;
        \?) echo "Valid options:"
            echo "-r : resize images to especified value."
            echo "-l : path where the LMDB will be created. Mandatory."
            echo "-d : data root (where the images live). Note that this is concatenated with the paths in the file list." 
            echo "-f : file containing a list of images and classes separated by space." 
            echo "Example:"
            echo "$0 -r 227 -l ../datasets/Imagenet_lmdb_train -d ../data/images/imagenet -f ../data/paths/imagenet_train.txt" 
            exit 0
        ;;
    esac
done

re=^[0-9]+$
if [ -z "$resize" ]; then resize=256; fi
if [ -z "$lmdb_path" ]; then echo "You have to provide the path where the LMDB will be created."; exit 1; fi
if [ -z "$data_root" ]; then echo "You have to provide the path where the images live."; exit 1; fi
if [ -z "$imgs_list" ]; then echo "You have to provide the file containing the list of images and classes"; exit 1; fi

if [[ ! "resize"=$re ]];
then
    echo "The resize argument (-r) has to be a number." >&2; exit 1;
fi

echo "Creating LMDB in $lmdb_path"
echo "Resizing images to $resize x $resize"

TOOLS=/home/$USER/.Software/caffe/build/tools

if [ ! -d "$data_root" ]; then
  echo "Error: $data_root is not a path to a directory."
  exit 1
fi

echo "Creating lmdb $lmdb_path"

GLOG_logtostderr=1 $TOOLS/convert_imageset  --resize_height=$resize  --resize_width=$resize --shuffle  $data_root  $imgs_list $lmdb_path

echo "Done."
