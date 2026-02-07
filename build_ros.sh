
echo "Building ROS nodes"

cd /YOLO_ORB_SLAM3/catkin_ws

source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
