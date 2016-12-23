#ifndef _MNIST_UTILS_
#define _MNIST_UTILS_
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <iostream>
#include <vector>

/*
 * This small module has utilities to parse the MNIST dataset (data and labels)
 * and return a vector of OpenCV Mats or a vector of labels (unsigned char)
 * Author: Ezequiel Torti Lopez
 */

using namespace std;
using namespace cv;

vector<Mat> load_images(string path);
vector<unsigned char> load_labels(string path);
#endif
