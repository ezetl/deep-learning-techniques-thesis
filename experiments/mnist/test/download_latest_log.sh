#!/usr/bin/env bash

# Download the latest Caffe log for plotting/processing/etc
# Change the SERVER name and LOGS FOLDER as needed
if [[ -z "$1" ]]; then
    echo ""
    echo "You have to call the script with path where you want to download the log:";
    echo "Example: $0 ./logs";
    echo ""
    exit 1;
fi
         
DOWNLOAD_DIR=$1
SERVER="ezetl@mini.famaf.unc.edu.ar"
LOGS_FOLDER="~/tesis/experiments/mnist/logs"
LOG_NAME=$(ssh $SERVER 'ls -t '$LOGS_FOLDER' | head -1')
OUT_FILE="caffe.log"
scp $SERVER:$LOGS_FOLDER/$LOG_NAME $DOWNLOAD_DIR/$OUT_FILE
echo "$DOWNLOAD_DIR/$OUT_FILE"
