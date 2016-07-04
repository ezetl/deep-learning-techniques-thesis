#!/usr/bin/env bash

# Download the latest Caffe snapshot for testing 
# Change the SERVER name and snapshots folder as needed
if [[ -z "$1" ]]; then
    echo ""
    echo "You have to call the script with path where you want to download the snapshot:";
    echo "Example: $0 ./models";
    echo ""
    exit 1;
fi
         
DOWNLOAD_DIR=$1
SERVER="ezetl@mini.famaf.unc.edu.ar"
LOGS_FOLDER="~/tesis/experiments/mnist/logs"
SNAPSHOT_NAME=$(ssh $SERVER 'ls -t '$SNAPSHOT_FOLDER' | head -2 | grep caffemodel')
scp $SERVER:$SNAPSHOT_FOLDER/$SNAPSHOT_NAME $DOWNLOAD_DIR 
echo "$DOWNLOAD_DIR/$SNAPSHOT_NAME"
