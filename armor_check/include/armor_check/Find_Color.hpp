#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

class Find_Color {
private:
  int hmin, smin, vmin;
  int hmax, smax, vmax;

public:
  void choose_hsv(Mat img);
};
