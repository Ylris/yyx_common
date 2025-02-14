#include <chrono>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

using namespace cv;
using namespace std::chrono_literals;
using std::placeholders::_1;

struct DetectionConfig {
  struct ColorRange {
    cv::Scalar lower;
    cv::Scalar upper;
    std::string name;
  };
  
  ColorRange blue_armor{{84,0,185}, {179,255,255}, "Blue"};
  ColorRange red_armor{{0,0,234}, {88,197,255}, "Red"};
  ColorRange number{{59,28,70}, {118,135,133}, "Num"};
  float gamma = 1.2;
  int min_area = 50;
};

class Visualizer {
public:
  static void draw_armor(cv::Mat& frame, cv::Rect rect, const std::string& label) {
    const int thickness = 3;
    const cv::Scalar main_color(0, 255, 255);
    const cv::Scalar text_color(255, 255, 255);
    
    cv::rectangle(frame, rect, main_color, thickness);
    cv::line(frame, rect.tl(), rect.tl() + cv::Point(0, -20), main_color, thickness);
    cv::putText(frame, label, rect.tl() + cv::Point(5, -5), 
               cv::FONT_HERSHEY_DUPLEX, 0.8, text_color, 2);
  }
};

class ArmorDetector {
public:
  explicit ArmorDetector(const DetectionConfig& cfg) : cfg_(cfg) {}

  cv::Rect process(cv::Mat& frame) {
    cv::Mat processed = apply_gamma(frame);
    return detect_armor(processed);
  }

private:
  cv::Mat apply_gamma(cv::Mat img) {
    if(cfg_.gamma == 1.0f) return img;
    cv::Mat lut(1, 256, CV_8U);
    uchar* p = lut.ptr();
    for(int i=0; i<256; ++i) 
      p[i] = cv::saturate_cast<uchar>(pow(i/255.0, cfg_.gamma)*255);
    cv::LUT(img, lut, img);
    return img;
  }

  cv::Rect detect_armor(cv::Mat& frame) {
    std::vector<std::vector<cv::Point> > color_ctrs = this->find_color(frame, cfg_.blue_armor);
    std::vector<std::vector<cv::Point> > num_ctrs = this->find_color(frame, cfg_.number);
    
    for(auto& n_ctr : num_ctrs) {
      cv::Rect num_rect = cv::boundingRect(n_ctr);
      if(num_rect.area() < 100) continue;
      
      for(auto& c1 : color_ctrs) {
        for(auto& c2 : color_ctrs) {
          cv::Rect r1 = cv::boundingRect(c1);
          cv::Rect r2 = cv::boundingRect(c2);
          if(abs(r1.x - num_rect.x) < 50 && abs(r2.x - num_rect.x) < 50) {
            return cv::Rect(
              std::min(r1.x, r2.x), 
              std::min(r1.y, r2.y),
              std::max(r1.br().x, r2.br().x) - std::min(r1.x, r2.x),
              std::max(r1.br().y, r2.br().y) - std::min(r1.y, r2.y)
            );
          }
        }
      }
    }
    return cv::Rect();
  }

  std::vector<std::vector<cv::Point> > find_color(cv::Mat& frame, const DetectionConfig::ColorRange& color) {
    cv::Mat hsv, mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, color.lower, color.upper, mask);
    
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    std::vector<std::vector<cv::Point> > filtered;
    for(auto& c : contours) {
      if(cv::contourArea(c) > cfg_.min_area)
        filtered.push_back(c);
    }
    return filtered;
  }

  DetectionConfig cfg_;
};

class ArmorNode : public rclcpp::Node {
public:
  ArmorNode() : Node("armor_node") {
    detector_ = std::make_unique<ArmorDetector>(DetectionConfig{});
    publisher_ = create_publisher<std_msgs::msg::String>("armor_color", 10);
    
    cap_.open("/home/amber/armor_check/resource/test.mp4");
    timer_ = create_wall_timer(50ms, std::bind(&ArmorNode::update, this));
  }

private:
  void update() {
    cv::Mat frame;
    if(!cap_.read(frame)) {
      RCLCPP_ERROR(get_logger(), "End of video");
      rclcpp::shutdown();
      return;
    }

    cv::Rect armor = detector_->process(frame);
    std_msgs::msg::String msg;
    msg.data = armor.area() ? "Detected" : "None";
    publisher_->publish(msg);

    if(armor.area()) 
      Visualizer::draw_armor(frame, armor, msg.data);
    
    cv::imshow("Armor Detection", frame);
    cv::waitKey(1);
  }

  cv::VideoCapture cap_;
  std::unique_ptr<ArmorDetector> detector_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArmorNode>());
  rclcpp::shutdown();
  return 0;
}