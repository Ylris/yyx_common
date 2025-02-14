#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct TargetColor {
  string name;
  Scalar lower;
  Scalar upper;
};

class ColorDetector {
private:
  /* data */
public:
  vector<vector<Point>> Detectcolor(Mat, TargetColor); //寻找单种颜色
  Rect Find_Armor(Mat &, TargetColor, TargetColor, TargetColor); //寻找装甲板
  void Find_TargetColorContours(Mat, vector<vector<Point>> &);
  void Show_TargetColorRect(Mat &, Rect, TargetColor);
  ColorDetector();
  ~ColorDetector();
};
