#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <mutex>
#include <chrono>
#include <string>

class ArmorDetectionVisualizer : public rclcpp::Node {
public:
  ArmorDetectionVisualizer() : Node("armor_detection_visualizer") {
    // 参数声明和加载
    declare_parameters();
    load_parameters();

    // 初始化视频流
    initialize_video();

    // 创建订阅者
    initialize_subscribers();

    // 创建定时器
    initialize_timer();

    // 初始化时间戳
    last_update_ = std::chrono::steady_clock::now();
    last_frame_time_ = std::chrono::steady_clock::now();
  }

  ~ArmorDetectionVisualizer() {
    if (cap_.isOpened()) {
      cap_.release();
    }
    cv::destroyAllWindows();
  }

private:
  void declare_parameters() {
    this->declare_parameter("video_path", "/home/amber/armor_check/resource/test.mp4");
    this->declare_parameter("font_scale", 0.8);
    this->declare_parameter("status_color", std::vector<int>{0, 255, 0});    // BGR格式
    this->declare_parameter("warning_color", std::vector<int>{0, 0, 255});
    this->declare_parameter("info_color", std::vector<int>{0, 255, 255});
    this->declare_parameter("bg_alpha", 0.7);
  }

  void load_parameters() {
    video_path_ = this->get_parameter("video_path").as_string();
    font_scale_ = this->get_parameter("font_scale").as_double();
    bg_alpha_ = this->get_parameter("bg_alpha").as_double();

    auto load_color = [&](const std::string& param_name) -> cv::Scalar {
      auto color_vec = this->get_parameter(param_name).as_integer_array();
      return cv::Scalar(color_vec[0], color_vec[1], color_vec[2]);
    };

    status_color_ = load_color("status_color");
    warning_color_ = load_color("warning_color");
    info_color_ = load_color("info_color");
  }

  void initialize_video() {
    cap_.open(video_path_);
    if (!cap_.isOpened()) {
      RCLCPP_ERROR(this->get_logger(), "无法打开视频文件: %s", video_path_.c_str());
      rclcpp::shutdown();
    }
  }

  void initialize_subscribers() {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));

    // 装甲板信息订阅
    armor_sub_ = this->create_subscription<std_msgs::msg::String>(
      "armor_info",
      qos,
      [this](const std_msgs::msg::String::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        armor_info_ = msg->data;
        last_update_ = std::chrono::steady_clock::now();
      });

    // 系统状态订阅
    system_sub_ = this->create_subscription<std_msgs::msg::String>(
      "system_status",
      qos,
      [this](const std_msgs::msg::String::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        system_status_ = msg->data;
      });
  }

  void initialize_timer() {
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(33),  // 约30FPS
      [this]() { process_frame(); });
  }

  void process_frame() {
    cv::Mat frame;
    if (!cap_.read(frame)) {
      RCLCPP_INFO(this->get_logger(), "视频播放完成");
      rclcpp::shutdown();
      return;
    }

    // 计算帧率
    auto now = std::chrono::steady_clock::now();
    double fps = 1e9 / std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_frame_time_).count();
    last_frame_time_ = now;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      draw_status_panel(frame, fps);
      draw_armor_info(frame);
    }

    cv::imshow("Armor Detection System", frame);
    handle_key_input();
  }

  void draw_status_panel(cv::Mat& frame, double fps) {
    const int panel_width = 300;
    const int margin = 20;
    
    // 绘制半透明背景
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, 
                 cv::Rect(margin, margin, panel_width, 140),
                 cv::Scalar(30, 30, 30),  // 深灰色
                 cv::FILLED);
    cv::addWeighted(overlay, bg_alpha_, frame, 1 - bg_alpha_, 0, frame);

    // 状态指示灯
    cv::Scalar status_color = (system_status_ == "NORMAL") ? status_color_ : warning_color_;
    cv::circle(frame, cv::Point(margin + 15, margin + 35), 8, status_color, -1);

    // 状态信息
    std::vector<std::pair<std::string, std::string>> status_info = {
      {"FPS", cv::format("%.1f", fps)},
      {"Status", system_status_},
      {"Update", format_duration() + " ago"}
    };

    cv::Point text_pos(margin + 40, margin + 35);
    for (const auto& [label, value] : status_info) {
      cv::putText(frame, label + ":", text_pos,
                 cv::FONT_HERSHEY_SIMPLEX, font_scale_ * 0.7,
                 info_color_, 1);
      cv::putText(frame, value, 
                 text_pos + cv::Point(100, 0),
                 cv::FONT_HERSHEY_SIMPLEX, font_scale_ * 0.7,
                 status_color_, 1);
      text_pos.y += 30;
    }
  }

  void draw_armor_info(cv::Mat& frame) {
    const int margin = 20;
    const int panel_width = 300;
    cv::Point pos(frame.cols - panel_width - margin, margin);

    // 动态呼吸效果
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - last_update_).count() / 1000.0;
    double alpha = 0.5 * (1.0 + sin(elapsed * 3.1416 * 2.0 / 2.0));  // 2秒周期

    // 绘制信息框
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, 
                 cv::Rect(pos.x, pos.y, panel_width, 80),
                 cv::Scalar(30, 30, 30),  // 深灰色
                 cv::FILLED);
    cv::addWeighted(overlay, bg_alpha_, frame, 1 - bg_alpha_, 0, frame);

    // 绘制文本
    cv::putText(frame, "DETECTED ARMOR:", 
               pos + cv::Point(15, 25),
               cv::FONT_HERSHEY_SIMPLEX, font_scale_ * 0.8,
               info_color_, 1);
    cv::putText(frame, armor_info_,
               pos + cv::Point(15, 55),
               cv::FONT_HERSHEY_DUPLEX, font_scale_,
               info_color_ * (0.7 + 0.3 * alpha), 2);
  }

  std::string format_duration() {
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now() - last_update_);
    return std::to_string(duration.count()) + "s";
  }

  void handle_key_input() {
    int key = cv::waitKey(1);
    if (key == 27 || key == 'q') {  // ESC或Q退出
      rclcpp::shutdown();
    }
  }

  // 成员变量
  cv::VideoCapture cap_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr armor_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr system_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::string video_path_;
  std::string armor_info_ = "WAITING...";
  std::string system_status_ = "INITIALIZING";
  
  cv::Scalar status_color_;
  cv::Scalar warning_color_;
  cv::Scalar info_color_;
  double font_scale_;
  double bg_alpha_;
  
  std::mutex mutex_;
  std::chrono::steady_clock::time_point last_update_;
  std::chrono::steady_clock::time_point last_frame_time_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArmorDetectionVisualizer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}