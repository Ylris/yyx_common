#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

class TraditionalArmorDetector : public rclcpp::Node
{
public:
    TraditionalArmorDetector() : Node("traditional_armor_detector")
    {

        video_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/video_topic", 10, std::bind(&TraditionalArmorDetector::video_callback, this, std::placeholders::_1)
        );
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr video_subscriber_;

    void video_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {

        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat frame = cv_ptr->image;

        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        cv::Mat mask;
        cv::Scalar lower_blue(100, 150, 50);   
        cv::Scalar upper_blue(140, 255, 255);  
        cv::inRange(hsv, lower_blue, upper_blue, mask);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (size_t i = 0; i < contours.size(); i++)
        {
            cv::Rect bounding_rect = cv::boundingRect(contours[i]);
            if (bounding_rect.area() > 500)  
            {

                cv::rectangle(frame, bounding_rect, cv::Scalar(0, 255, 0), 2);


                cv::Scalar color = get_armor_color(frame(bounding_rect));

                cv::putText(frame, "Color: Blue", cv::Point(bounding_rect.x, bounding_rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
                cv::putText(frame, "ID: 123", cv::Point(bounding_rect.x + bounding_rect.width - 60, bounding_rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);


                detect_number(frame(bounding_rect));
            }
        }

        cv::imshow("Traditional Armor Detection", frame);
        cv::waitKey(1);
    }

    cv::Scalar get_armor_color(const cv::Mat& region)
    {
        cv::Mat hsv_region;
        cv::cvtColor(region, hsv_region, cv::COLOR_BGR2HSV);

        cv::Scalar mean_hsv = cv::mean(hsv_region);

        return cv::Scalar(0, 0, 255); 
    }


    void detect_number(cv::Mat& image)
    {
        std::vector<cv::Mat> templates;

        for (int i = 0; i < 10; i++) {
            templates.push_back(cv::imread("digit_templates/" + std::to_string(i) + ".png", cv::IMREAD_GRAYSCALE));
        }

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);  

        for (int i = 0; i < 10; i++) {

            cv::Mat template_img = templates[i];
            int result_cols = gray.cols - template_img.cols + 1;
            int result_rows = gray.rows - template_img.rows + 1;

            cv::Mat result;
            result.create(result_rows, result_cols, CV_32FC1);

            cv::matchTemplate(gray, template_img, result, cv::TM_CCOEFF_NORMED);

            double min_val, max_val;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

            if (max_val > 0.8) {  
                cv::rectangle(image, max_loc, cv::Point(max_loc.x + template_img.cols, max_loc.y + template_img.rows), cv::Scalar(0, 255, 0), 2);
                cv::putText(image, std::to_string(i), cv::Point(max_loc.x, max_loc.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
            }
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TraditionalArmorDetector>());
    rclcpp::shutdown();
    return 0;
}

