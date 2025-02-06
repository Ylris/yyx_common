#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>

class VideoPublisher : public rclcpp::Node {
public:
  VideoPublisher() : Node("video_publisher_node") {

    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("video_frames", 10);
    
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(33),
      std::bind(&VideoPublisher::timer_callback, this));

    cap.open("test_video.mp4");
    if (!cap.isOpened()) {
      RCLCPP_ERROR(this->get_logger(), "无法打开视频文件！");
      rclcpp::shutdown();
    }
  }

private:
  void timer_callback() {
  cv::Mat frame;
  if (cap.read(frame)) {

    cv::imshow("Video Publisher", frame);
    cv::waitKey(1);  
    
    auto msg = cv_bridge::CvImage(
      std_msgs::msg::Header(), "bgr8", frame
    ).toImageMsg();
    publisher_->publish(*msg);
    RCLCPP_INFO(this->get_logger(), "发布视频帧");
  }
}

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  cv::VideoCapture cap;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VideoPublisher>());
  rclcpp::shutdown();
  return 0;
}
