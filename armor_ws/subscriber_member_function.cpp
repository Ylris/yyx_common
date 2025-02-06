#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/msg/image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "std_msgs/msg/string.hpp"
using std::placeholders::_1;

using namespace cv;
using namespace std;

// 用于识别算法的变量
const int kThreashold = 220;
const int kMaxVal = 255;
const Size kGaussianBlueSize = Size(5, 5);
Mat frame,channels[3],binary,Gaussian;
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;
Rect boundRect;
RotatedRect box;
vector<Point2f> boxPts(4);

// 用于写入视频
// Size videoSize(1280,960);
// int fps = 30;
// VideoWriter writer("/home/wh/test.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, videoSize, true);

class MinimalSubscriber : public rclcpp::Node
{
  public:
    MinimalSubscriber()
    : Node("minimal_subscriber")
    {

      rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
      sub_ = image_transport::create_subscription(this, "/color/image_raw",
                std::bind(&MinimalSubscriber::topic_callback, this, std::placeholders::_1), "raw", custom_qos);
      cv::namedWindow(OPENCV_WINDOW);
    }
  private:
    void topic_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) const
    {
      cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // cv_ptr就是转换后的结果，调用cv_ptr -> image即可获得cv：：mat数据
        // cv::imshow(OPENCV_WINDOW, cv_ptr->image);
        // 这里将每一帧图像用作应用识别算法
        cv::Mat old_mat = cv_ptr->image;
        cv::Mat new_mat = dealing_image(old_mat);
        cv::imshow(OPENCV_WINDOW, new_mat);

        // 上面的代码都已经成功运行，接下来可以考虑将每一帧图像数据输出成视频了
        // writer.write(new_mat);
        cv::waitKey(3);
      
    }
    cv::Mat dealing_image(cv::Mat &img) const {
      Rect point_array[20];
      // 这里要有个深拷贝
      cv::Mat cp_img;
      img.copyTo(cp_img);
      split(cp_img,channels);
      threshold(channels[0], binary, kThreashold, kMaxVal, 0);
      GaussianBlur(binary, Gaussian, kGaussianBlueSize, 0);
      findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
      int index = 0;
      for (int i = 0; i < contours.size(); i++) {
          //box = minAreaRect(Mat(contours[i]));
          //box.points(boxPts.data());
          boundRect = boundingRect(Mat(contours[i]));
          //rectangle(frame, boundRect.tl(), boundRect.br(), (0, 255, 0), 2,8 ,0);

          try
          {
              if (double(boundRect.height / boundRect.width) >= 1.3 && boundRect.height > 36 && boundRect.height>20) {
                  point_array[index] = boundRect;
                  index++;
              }
          }
          catch (const char* msg)
          {
              // cout << printf(msg) << endl;
              //continue;
          }
      }        
      int point_near[2];
      int min = 10000;
      for (int i = 0; i < index-1; i++)
      {
          for (int j = i + 1; j < index; j++) {
              int value = abs(point_array[i].area() - point_array[j].area());
              if (value < min)
              {
                  min = value;
                  point_near[0] = i;
                  point_near[1] = j;
              }
          }
      }   
      try
      {
          Rect rectangle_1 = point_array[point_near[0]];
          Rect rectangle_2 = point_array[point_near[1]];
          if (rectangle_2.x == 0 || rectangle_1.x == 0) {
              throw "not enough points";
          }
          Point point1 = Point(rectangle_1.x + rectangle_1.width / 2, rectangle_1.y);
          Point point2 = Point(rectangle_1.x + rectangle_1.width / 2, rectangle_1.y + rectangle_1.height);
          Point point3 = Point(rectangle_2.x + rectangle_2.width / 2, rectangle_2.y);
          Point point4 = Point(rectangle_2.x + rectangle_2.width / 2, rectangle_2.y + rectangle_2.height);
          Point p[4] = { point1,point2,point4,point3 };
          cout << p[0]<<p[1]<<p[2]<<p[3] << endl;
          for (int i = 0; i < 4; i++) {
              line(cp_img, p[i%4], p[(i+1)%4], Scalar(0, 255, 0), 2);
          }           
      }
      catch (const char* msg)
      {
          cout << msg << endl;
          //continue;
      }

      return cp_img;
    }
    image_transport::Subscriber sub_;

    const std::string OPENCV_WINDOW = "Image window";
};



int main(int argc, char * argv[])
{

  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();

  // 关闭视频写入流
      // writer.release();
  return 0;
}