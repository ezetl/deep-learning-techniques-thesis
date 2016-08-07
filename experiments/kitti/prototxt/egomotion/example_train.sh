#!/usr/bin/env bash
GLOG_log_dir=$LOGS_KITTI_DIR caffe train -gpu=0 -solver=solver.prototxt
