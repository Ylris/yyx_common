#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <armor_detector/msg/armor_detection.hpp>  // YOLO 和传统视觉识别结果消息
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/approximate_time.h>

class DisplayNode : public rclcpp::Node
{
public:
    DisplayNode() : Node("display_node")
    {
        // 创建视频、YOLO和传统视觉结果的订阅者
        sub_video_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, "/video_topic");
        sub_dl_ = std::make_shared<message_filters::Subscriber<armor_detector::msg::ArmorDetection>>(this, "/yolo_results");
        sub_trad_ = std::make_shared<message_filters::Subscriber<armor_detector::msg::ArmorDetection>>(this, "/traditional_results");

        // 使用ApproximateTime策略同步订阅者
        sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, armor_detector::msg::ArmorDetection, armor_detector::msg::ArmorDetection>>( 
                    *sub_video_, *sub_dl_, *sub_trad_, 10);
        sync_->registerCallback(std::bind(&DisplayNode::sync_callback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    }

private:
    message_filters::Subscriber<sensor_msgs::msg::Image>::SharedPtr sub_video_;
    message_filters::Subscriber<armor_detector::msg::ArmorDetection>::SharedPtr sub_dl_;
    message_filters::Subscriber<armor_detector::msg::ArmorDetection>::SharedPtr sub_trad_;
    std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, armor_detector::msg::ArmorDetection, armor_detector::msg::ArmorDetection>> sync_;

    void sync_callback(const sensor_msgs::msg::Image::SharedPtr video_msg,
                       const armor_detector::msg::ArmorDetection::SharedPtr dl_msg,
                       const armor_detector::msg::ArmorDetection::SharedPtr trad_msg)
    {
        // 使用cv_bridge将ROS图像消息转换为OpenCV图像
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(video_msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // 获取视频帧
        cv::Mat frame = cv_ptr->image;

        // 在原始视频上叠加YOLO识别结果（如颜色）
        cv::putText(frame, "YOLO Color: " + dl_msg->color, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        // 叠加传统视觉识别结果（如数字）
        cv::putText(frame, "Traditional ID: " + std::to_string(trad_msg->id), cv::Point(30, 70), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);

        // 显示合成的视频帧
        cv::imshow("Combined Video", frame);
        cv::waitKey(1);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DisplayNode>());
    rclcpp::shutdown();
    return 0;
}

