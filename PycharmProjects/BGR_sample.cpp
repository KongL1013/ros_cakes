#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <string>
using namespace std;

int cnt=1;
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv::Mat imgae=cv_bridge::toCvShare(msg, "bgr8")->image;
    cv::imshow("view",imgae );
    if(cv::waitKey(30)==27){  //67=c
        std::ostringstream stm ;
        stm << cnt ;
        std::string nnn="/home/dd/ros_realsense2/src/get_rs_pic/pic/"+stm.str()+".jpg";
        cv::imwrite(nnn,imgae);
        cnt++;
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 1, imageCallback);
    ros::spin();

}
