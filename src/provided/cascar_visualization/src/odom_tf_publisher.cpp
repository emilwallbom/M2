#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_broadcaster.h"

#include <functional>
#include <cmath>
#include <memory>
#include <string>

using std::placeholders::_1;

class OdomTfPublisher : public rclcpp::Node {
public:
  OdomTfPublisher() : Node("odom_tf_publisher") {
    this->declare_parameter<std::string>("agent_name", "");
    const auto agent_name = this->get_parameter("agent_name").as_string();
    const auto prefix = agent_name.empty() ? "" : agent_name + "/";
    odom_frame_ = prefix + "odom";
    base_link_frame_ = prefix + "base_link";

    broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    static_broadcaster_ =
        std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
    path_publisher_ = this->create_publisher<nav_msgs::msg::Path>(
        "driven_path", rclcpp::QoS(1).transient_local());
    odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "odom", 1, std::bind(&OdomTfPublisher::odomCallback, this, _1));

    geometry_msgs::msg::TransformStamped map_to_odom;
    map_to_odom.header.stamp = this->get_clock()->now();
    map_to_odom.header.frame_id = "map";
    map_to_odom.child_frame_id = odom_frame_;
    map_to_odom.transform.rotation.w = 1.0;
    static_broadcaster_->sendTransform(map_to_odom);
  }

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr message) {
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = message->header.stamp;
    transform.header.frame_id = odom_frame_;
    transform.child_frame_id = base_link_frame_;
    transform.transform.translation.x = message->pose.pose.position.x;
    transform.transform.translation.y = message->pose.pose.position.y;
    transform.transform.translation.z = message->pose.pose.position.z;
    transform.transform.rotation = message->pose.pose.orientation;
    broadcaster_->sendTransform(transform);

    const auto frame = message->header.frame_id.empty() ? odom_frame_ : message->header.frame_id;
    // A frame change or clock reset starts a new trail.
    if (path_.header.frame_id != frame ||
        rclcpp::Time(message->header.stamp) < rclcpp::Time(path_.header.stamp)) {
      path_.poses.clear();
    }
    path_.header = message->header;
    path_.header.frame_id = frame;
    if (!path_.poses.empty()) {
      const auto & previous = path_.poses.back().pose.position;
      const auto & current = message->pose.pose.position;
      if (std::hypot(current.x - previous.x, current.y - previous.y) < 0.02) {
        return;
      }
    }
    geometry_msgs::msg::PoseStamped pose;
    pose.header = path_.header;
    pose.pose = message->pose.pose;
    // Retain approximately 100 m at the minimum sample spacing.
    if (path_.poses.size() >= 5000) {
      path_.poses.erase(path_.poses.begin());
    }
    path_.poses.push_back(pose);
    path_publisher_->publish(path_);
  }

  std::string odom_frame_;
  std::string base_link_frame_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
  nav_msgs::msg::Path path_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OdomTfPublisher>());
  rclcpp::shutdown();
  return 0;
}
