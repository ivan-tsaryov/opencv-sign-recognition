#include "Recognition.h"

#define FORWARD_CODE 1320   // 0002550255002550255  - 10100101000
#define RIGHT_CODE  176     // 00002552550255000    - 00010110000
#define LEFT_CODE  608      // 00000255255002550    - 01001100000
#define STOP_CODE  112      // 00002552552550000    - 00001110000
#define CODE_LENGTH 11  

#define CONTOUR_THRESHOLD 2000

#define AREA_DELTA 0.055   // Area size
#define AREA_THRESHOLD 40  // Threshold for white pixels of area

RNG rng(12345);

int alpha = 2, beta = 140; // For circle/ellipse detection
int alpha_src = 2, beta_src = 140; // For source image
int alpha_MAX = 3; // Contrast limit
int beta_MAX = 300; // Brightness limit

// For source image
int thresh_src = 100;
int max_thresh_src = 255;

// For circle/ellipse detection
int thresh = 255;
int max_thresh = 255;

int ksize = 9;

int Recognition::calc_area_color(Mat img, double centerY, double centerX) {
	int x1 = (int)round((centerX - AREA_DELTA) * img.cols);
	int y1 = (int)round((centerY - AREA_DELTA) * img.rows);
	int x2 = (int)round((centerX + AREA_DELTA) * img.cols);
	int y2 = (int)round((centerY + AREA_DELTA) * img.rows);

	Rect roi = Rect(Point(x1, y1), Point(x2, y2));
	int area = countNonZero(img(roi));

	if (area > AREA_THRESHOLD)
		return 255;
	else
		return 0;
}

int Recognition::calc_bin_similarity(int code, int *tmpl, int size) {
	int result = 0;
	int max = 0;
	for (int j = 0; j < size; j++) {
		int buf = 1;
		int count = 0;
		for (int i = 0; i < CODE_LENGTH; i++) {
			if ((code & buf) == (tmpl[j] & buf))
				count++;
			buf *= 2;
		}
		if (count > max && count >(CODE_LENGTH - 3)) {
			max = count;
			result = tmpl[j];
		}
	}
	return result;
}

string Recognition::detect_sign(Mat img) {
	img.convertTo(img, -1, alpha_src, beta_src - 200);

	ksize = img.size().width / 15;
	if (ksize % 2 == 0) {
		ksize++;
	}

	int top_left = 0, top = 0, top_right = 0, top_middle = 0;
	int left = 0, center = 0, right = 0;
	int bot_left = 0, bot = 0, bot_right = 0, bot_middle = 0;

	top_left = calc_area_color(img, 0.30, 0.25);
	top = calc_area_color(img, 0.15, 0.50);
	top_right = calc_area_color(img, 0.30, 0.75);
	top_middle = calc_area_color(img, 0.30, 0.50);
	left = calc_area_color(img, 0.50, 0.25);
	center = calc_area_color(img, 0.50, 0.50);
	right = calc_area_color(img, 0.50, 0.75);
	bot_left = calc_area_color(img, 0.70, 0.25);
	bot = calc_area_color(img, 0.85, 0.50);
	bot_right = calc_area_color(img, 0.70, 0.75);
	bot_middle = calc_area_color(img, 0.70, 0.50);

	int res = 0;
	res |= top_left ? 1 : 0;
	res |= top ? 2 : 0;
	res |= top_right ? 4 : 0;
	res |= top_middle ? 8 : 0;
	res |= left ? 16 : 0;
	res |= center ? 32 : 0;
	res |= right ? 64 : 0;
	res |= bot_left ? 128 : 0;
	res |= bot ? 256 : 0;
	res |= bot_right ? 512 : 0;
	res |= bot_middle ? 1024 : 0;

	int sim_arr[4] = { FORWARD_CODE, RIGHT_CODE, LEFT_CODE, STOP_CODE };
	int sim_code = calc_bin_similarity(res, sim_arr, 4);
	switch (sim_code) {
	case FORWARD_CODE:
		return "This is forward";
	case RIGHT_CODE:
		return "This is right";
	case LEFT_CODE:
		return "This is left";
	case STOP_CODE:
		return "This is stop";
	default:
		return "Not recognized";
	}
}

// Set mask on blue and red colorspace
Mat Recognition::set_mask(Mat image) {
	Mat hsv_image, result;

	cvtColor(image, hsv_image, CV_BGR2HSV);

	Mat lower_red_hue_range;
	Mat upper_red_hue_range;
	Mat blue_hue_range;

	inRange(hsv_image, Scalar(0, 100, 100), Scalar(10, 255, 255), lower_red_hue_range);
	inRange(hsv_image, Scalar(160, 100, 100), Scalar(179, 255, 255), upper_red_hue_range);
	inRange(hsv_image, Scalar(100, 100, 50), Scalar(140, 255, 255), blue_hue_range);

	addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, result);
	addWeighted(result, 1.0, blue_hue_range, 1.0, 0.0, result);

	//morphological opening (remove small objects from the foreground)
	/*erode(result, result, getStructuringElement(MORPH_ELLIPSE, Size(FILTER_AREA_SIZE, FILTER_AREA_SIZE)));
	dilate(result, result, getStructuringElement(MORPH_ELLIPSE, Size(30, 30)));*/

	//morphological closing (fill small holes in the foreground)
	//dilate(result, result, getStructuringElement(MORPH_ELLIPSE, Size(FILTER_AREA_SIZE, FILTER_AREA_SIZE)));
	//erode(result, result, getStructuringElement(MORPH_ELLIPSE, Size(FILTER_AREA_SIZE, FILTER_AREA_SIZE)));
	//imshow("Bla", result);

	return result;
}

// Calculate white-black pixels ratio
float Recognition::wb_ratio(Mat img) {
	int x1 = (int)round(0.1*img.cols);
	int y1 = (int)round(0.1*img.rows);
	int x2 = (int)round(0.9*img.cols);
	int y2 = (int)round(0.9*img.rows);

	Rect roi = Rect(Point(x1, y1), Point(x2, y2));
	img = img(roi);

	return ((float)countNonZero(img) / (float)img.total());
}

// Autocorrect threshold level
void Recognition::threshold_autocorrection(Mat img) {
	Mat copy = img.clone();
	copy = copy > thresh_src;

	if (wb_ratio(copy) > 0.25) {
		do {
			copy = img.clone();
			thresh_src++;
			copy = copy > thresh_src;
			setTrackbarPos("Src Thresh", "Webcam", thresh_src);
		} while (wb_ratio(copy) > 0.2);
	} else if (wb_ratio(copy) < 0.3) {
		do {
			copy = img.clone();
			thresh_src--;
			copy = copy > thresh_src;
			setTrackbarPos("Src Thresh", "Webcam", thresh_src);
		} while (wb_ratio(copy) < 0.2);
	}
}

void Recognition::detectSign(VideoCapture cap, bool need_show_window, int capture_count) {
	if (!cap.isOpened()) {
		cout << "Cannot open source cam" << endl;
		return;
	}

	if (need_show_window) {
		namedWindow("Webcam", CV_WINDOW_AUTOSIZE);
		namedWindow("Founded shape", CV_WINDOW_AUTOSIZE);

		createTrackbar("Contr", "Webcam", &alpha, alpha_MAX);
		createTrackbar("Src Contr", "Webcam", &alpha_src, alpha_MAX);
		createTrackbar("Bright", "Webcam", &beta, beta_MAX);
		createTrackbar("Src Bright", "Webcam", &beta_src, beta_MAX);
		createTrackbar("Thresh", "Webcam", &thresh, max_thresh);
		createTrackbar("Src Thresh", "Webcam", &thresh_src, max_thresh_src);
	}
	Mat src, blue_and_red, orig_image, canny_output, drawing;
	long k = 0;

	while (true) {
		cap >> src;
		src.copyTo(orig_image);

		// Contrast changing (f = alpha*img + beta)
		src.convertTo(src, -1, alpha, beta - 200);
		// Get only blue and red colors from src image
		blue_and_red = set_mask(src);
		// Invert image
		bitwise_not(blue_and_red, blue_and_red);

		equalizeHist(blue_and_red, blue_and_red);
		// Reduce the noise so we avoid false circle/ellipse detection
		medianBlur(blue_and_red, blue_and_red, ksize);

		// Find contours from binary image  
		vector< vector<Point> > contours;
		vector<Vec4i> hierarchy;

		Canny(blue_and_red, canny_output, thresh, thresh * 2, 3);

		findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));

		vector<vector<Point> > contours_poly(contours.size());
		vector<RotatedRect> minRect(contours.size());
		vector<RotatedRect> minEllipse(contours.size());
		vector<Rect> boundRect(contours.size());
		vector<float>radius(contours.size());
		vector<float>area(contours.size());
		vector<Point2f>center(contours.size());
		drawing = Mat::zeros(blue_and_red.size(), CV_8UC3);

		for (size_t i = 0; i < contours.size(); i++) {
			approxPolyDP(Mat(contours[i]), contours_poly[i], 10, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			minEnclosingCircle(contours_poly[i], center[i], radius[i]);
			area[i] = (float)contourArea(Mat(contours_poly[i]));

			if (area[i] < CONTOUR_THRESHOLD) continue;

			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
			ellipse(drawing, minEllipse[i], color, 2, 8);

			Mat detected_area = orig_image(boundRect[i]);
			cvtColor(detected_area, detected_area, CV_BGR2GRAY);

			threshold_autocorrection(detected_area);

			detected_area = detected_area > thresh_src;


			if (need_show_window) {
				imshow("Founded shape", detected_area);
			}
			string res = detect_sign(detected_area);
			cout << res << endl;

			detected_area.release();
		}

		if (need_show_window) {
			imshow("Webcam", drawing);
		}

		if (k >= capture_count && capture_count > 0)
			break;
		if (k < capture_count)
			k++;
		if (waitKey(30) == 27)
			break;
	}
}