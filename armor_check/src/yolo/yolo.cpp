#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <iomanip>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <sstream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

// 常量定义
constexpr int kBorderThickness = 2;
constexpr float kConfidenceThreshold = 0.3f;
constexpr float kNmsThreshold = 0.4f;
constexpr float kFontScale = 0.7f;
constexpr int kFontThickness = 2;
const Scalar kBlueColor(255, 144, 30);    // BGR格式
const Scalar kRedColor(30, 30, 255);      // BGR格式
const Scalar kTextBgColor(50, 50, 50);    // 文字背景色
const Scalar kFpsColor(0, 255, 0);        // FPS显示颜色
const Size kModelInputSize = {256, 256};
const vector<string> kClassLabels = {"B1", "B2", "B3", "B4", "B5", "B7",
                                    "R1", "R2", "R3", "R4", "R5", "R7"};

class YoloNode : public rclcpp::Node {
public:
  YoloNode() : Node("yolo_node"), last_tick_count_(getTickCount()) {
    // 初始化ROS组件
    label_publisher_ = create_publisher<std_msgs::msg::String>("armor_detection", 10);
    
    // 初始化视频和模型
    initialize_model("/home/amber/armor_check/best.onnx");
    initialize_video("/home/amber/armor_check/src/armor_check/resource/test.mp4");
    
    // 创建定时器
    timer_ = create_wall_timer(50ms, bind(&YoloNode::process_video, this));
  }

private:
  void initialize_model(const string& model_path) {
    detection_model_ = readNetFromONNX(model_path);
    if (detection_model_.empty()) {
      RCLCPP_ERROR(get_logger(), "Failed to load model!");
      rclcpp::shutdown();
    }
  }

  void initialize_video(const string& video_path) {
    video_capture_.open(video_path);
    if (!video_capture_.isOpened()) {
      RCLCPP_ERROR(get_logger(), "Failed to open video!");
      rclcpp::shutdown();
    }
  }

  void process_video() {
    Mat frame;
    if (!video_capture_.read(frame)) {
      RCLCPP_WARN(get_logger(), "Video frame is empty!");
      rclcpp::shutdown();
      return;
    }

    // 预处理和推理
    Mat output = run_inference(frame);
    
    // 后处理
    vector<Rect> boxes;
    vector<float> confidences;
    vector<int> class_ids;
    process_detections(output, frame.size(), boxes, confidences, class_ids);

    // 应用NMS并绘制结果
    vector<int> indices;
    NMSBoxes(boxes, confidences, kConfidenceThreshold, kNmsThreshold, indices);
    draw_detections(frame, boxes, class_ids, confidences, indices);

    // 显示性能指标
    draw_fps_counter(frame);
    
    // 显示结果
    display_frame(frame);
  }

  Mat run_inference(Mat& frame) {
    // 图像预处理
    Mat blob = dnn::blobFromImage(frame, 1.0/255, kModelInputSize, Scalar(), true, false);
    detection_model_.setInput(blob);
    return detection_model_.forward();
  }

  void process_detections(const Mat& output, Size frame_size,
                         vector<Rect>& boxes, vector<float>& confidences,
                         vector<int>& class_ids) {
    Mat detections = output.reshape(1, output.size[1]);
    float scale_x = frame_size.width / static_cast<float>(kModelInputSize.width);
    float scale_y = frame_size.height / static_cast<float>(kModelInputSize.height);

    for (int i = 0; i < detections.cols; ++i) {
      Mat scores = detections.rowRange(4, detections.rows).col(i);
      Point class_id;
      double confidence;
      minMaxLoc(scores, nullptr, &confidence, nullptr, &class_id);

      if (confidence > kConfidenceThreshold) {
        float cx = detections.at<float>(0, i) * scale_x;
        float cy = detections.at<float>(1, i) * scale_y;
        float w = detections.at<float>(2, i) * scale_x;
        float h = detections.at<float>(3, i) * scale_y;

        int x = cvRound(cx - w * 0.5);
        int y = cvRound(cy - h * 0.5);
        boxes.emplace_back(x, y, cvRound(w), cvRound(h));
        confidences.push_back(confidence);
        class_ids.push_back(class_id.y);
      }
    }
  }

  void draw_detections(Mat& frame, const vector<Rect>& boxes,
                      const vector<int>& class_ids, const vector<float>& confidences,
                      const vector<int>& indices) {
    bool armor_detected = false;
    
    for (const int idx : indices) {
      const Rect& box = boxes[idx];
      const int class_id = class_ids[idx];
      const string label = kClassLabels[class_id];
      const float confidence = confidences[idx];
      armor_detected = true;

      // 根据队伍选择颜色
      Scalar color = (label[0] == 'B') ? kBlueColor : kRedColor;

      // 绘制带圆角的检测框
      rounded_rectangle(frame, box, color);

      // 创建带背景的文字标签
      ostringstream ss;
      ss << label << " " << fixed << setprecision(1) << confidence * 100 << "%";
      draw_text_label(frame, ss.str(), box.tl(), color);
    }

    // 发布检测结果
    publish_detection_status(armor_detected);
  }

  void rounded_rectangle(Mat& img, const Rect& box, const Scalar& color) {
    const int radius = 8;
    Point tl = box.tl();
    Point br = box.br();

    // 绘制圆角矩形
    line(img, Point(tl.x+radius, tl.y), Point(br.x-radius, tl.y), color, kBorderThickness);
    line(img, Point(tl.x+radius, br.y), Point(br.x-radius, br.y), color, kBorderThickness);
    line(img, Point(tl.x, tl.y+radius), Point(tl.x, br.y-radius), color, kBorderThickness);
    line(img, Point(br.x, tl.y+radius), Point(br.x, br.y-radius), color, kBorderThickness);

    // 绘制四个圆角
    ellipse(img, Point(tl.x+radius, tl.y+radius), Size(radius, radius), 180, 0, 90, color, kBorderThickness);
    ellipse(img, Point(br.x-radius, tl.y+radius), Size(radius, radius), 270, 0, 90, color, kBorderThickness);
    ellipse(img, Point(tl.x+radius, br.y-radius), Size(radius, radius), 90, 0, 90, color, kBorderThickness);
    ellipse(img, Point(br.x-radius, br.y-radius), Size(radius, radius), 0, 0, 90, color, kBorderThickness);
  }

  void draw_text_label(Mat& img, const string& text, Point position, const Scalar& color) {
    const int padding = 4;
    int baseline = 0;
    Size text_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, kFontScale, kFontThickness, &baseline);

    // 绘制文字背景
    Rect bg_rect(position.x, position.y - text_size.height - padding,
                text_size.width + padding*2, text_size.height + padding*2);
    rectangle(img, bg_rect, kTextBgColor, FILLED);

    // 绘制文字
    putText(img, text, position - Point(0, padding), FONT_HERSHEY_SIMPLEX,
           kFontScale, color, kFontThickness, LINE_AA);
  }

  void draw_fps_counter(Mat& frame) {
    // 计算FPS
    double current_time = static_cast<double>(getTickCount());
    double elapsed = (current_time - last_tick_count_) / getTickFrequency();
    last_tick_count_ = current_time;
    double fps = 1.0 / elapsed;

    // 绘制FPS显示
    ostringstream ss;
    ss << "FPS: " << fixed << setprecision(1) << fps;
    putText(frame, ss.str(), Point(20, 50), FONT_HERSHEY_SIMPLEX,
           kFontScale, kFpsColor, kFontThickness, LINE_AA);
  }

  void display_frame(const Mat& frame) {
    // 保持宽高比调整显示尺寸
    constexpr double kDisplayScale = 0.8;
    Size display_size(cvRound(frame.cols * kDisplayScale),
                     cvRound(frame.rows * kDisplayScale));
    
    Mat resized_frame;
    resize(frame, resized_frame, display_size, 0, 0, INTER_LINEAR);
    imshow("Armor Detection System", resized_frame);
    waitKey(1);
  }

  void publish_detection_status(bool armor_detected) {
    auto msg = std_msgs::msg::String();
    msg.data = armor_detected ? "armor_detected" : "no_armor";
    label_publisher_->publish(msg);
    RCLCPP_INFO_STREAM(get_logger(), "Detection status: " << msg.data);
  }

  // 成员变量
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr label_publisher_;
  VideoCapture video_capture_;
  Net detection_model_;
  rclcpp::TimerBase::SharedPtr timer_;
  int64 last_tick_count_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = make_shared<YoloNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}