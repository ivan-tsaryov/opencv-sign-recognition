#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

class Recognition {
private:
	int calc_area_color(Mat img, double centerY, double centerX);
	int calc_bin_similarity(int code, int *tmpl, int size);
	string detect_sign(Mat img);
	Mat set_mask(Mat image);
	float wb_ratio(Mat img);
	void threshold_autocorrection(Mat img);

public:
	void detectSign(VideoCapture cap, bool need_show_window, int capture_count);
};