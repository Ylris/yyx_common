#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "armor_detector/msg/ArmorDetection.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vector>

class YoloDetector : public rclcpp::Node {
public:
  YoloDetector() : Node("yolo_detector") {
    net_ = cv::dnn::readNetFromONNX("/home/amber/armor_wa/src/armor_detector/models/");
    
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/video_frames", 10,
      std::bind(&YoloDetector::callback, this, std::placeholders::_1));

    pub_ = create_publisher<armor_detector::msg::ArmorDetection>(
      "/armor_detections", 10);
  }

private:
  void callback(const sensor_msgs::msg::Image::SharedPtr msg) {

    cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;

    cv::Mat blob = cv::dnn::blobFromImage(
      frame, 1/255.0, cv::Size(640, 640), 
      cv::Scalar(), true, false, CV_32F
    );

    net_.setInput(blob);
    cv::Mat outputs = net_.forward();

    std::vector<cv::Rect> boxes = post_process(outputs, frame.size());

    for (const auto& box : boxes) {
      draw_detection(frame, box);
      publish_detection(msg->header, box);
    }

    cv::imshow("YOLOv5 Detection", frame);
    cv::waitKey(1);
  }

  std::vector<cv::Rect> post_process(cv::Mat& outputs, const cv::Size& frame_size) {
    std::vector<cv::Rect> boxes;
    const float* data = (float*)outputs.data;
    const int dimensions = 85;  
    const int rows = outputs.size[1];

    for(int i = 0; i < rows; ++i) {
      float confidence = data[4];
      if(confidence > 0.5) {
        float cx = data[0] * frame_size.width;
        float cy = data[1] * frame_size.height;
        float w = data[2] * frame_size.width;
        float h = data[3] * frame_size.height;

        boxes.emplace_back(cx - w/2, cy - h/2, w, h);
      }
      data += dimensions;
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, std::vector<float>(boxes.size(), 1.0), 
                      0.5, 0.4, indices);

    std::vector<cv::Rect> filtered_boxes;
    for(int idx : indices) {
      filtered_boxes.push_back(boxes[idx]);
    }
    return filtered_boxes;
  }

  void draw_detection(cv::Mat& frame, const cv::Rect& box) {
    cv::rectangle(frame, box, cv::Scalar(0,255,0), 2);
    cv::putText(frame, "Armor", cv::Point(box.x, box.y-5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
  }

  void publish_detection(const std_msgs::msg::Header& header, 
                        const cv::Rect& box) {
    auto msg = armor_detector::msg::ArmorDetection();
    msg.header = header;
    msg.x = box.x;
    msg.y = box.y;
    msg.width = box.width;
    msg.height = box.height;
    pub_->publish(msg);
  }

  cv::dnn::Net net_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<armor_detector::msg::ArmorDetection>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<YoloDetector>());
  rclcpp::shutdown();
  return 0;
}
