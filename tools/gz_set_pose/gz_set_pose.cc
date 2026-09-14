// Persistent Gazebo pose client: reads "name x y z qx qy qz qw" lines from stdin and calls the
// world's set_pose service for each. Used by uav_dt_ros/follow_cam_node to move the video camera
// at 10 Hz without paying the gz CLI start-up cost per call.
//   gz_set_pose <world>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/transport/Node.hh>

#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char **argv) {
  if (argc < 2) { std::cerr << "usage: gz_set_pose <world>\n"; return 1; }
  const std::string service = std::string("/world/") + argv[1] + "/set_pose";
  gz::transport::Node node;
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string name; double x, y, z, qx, qy, qz, qw;
    if (!(in >> name >> x >> y >> z >> qx >> qy >> qz >> qw)) continue;
    gz::msgs::Pose req;
    req.set_name(name);
    req.mutable_position()->set_x(x); req.mutable_position()->set_y(y); req.mutable_position()->set_z(z);
    req.mutable_orientation()->set_x(qx); req.mutable_orientation()->set_y(qy);
    req.mutable_orientation()->set_z(qz); req.mutable_orientation()->set_w(qw);
    gz::msgs::Boolean rep; bool result = false;
    if (!node.Request(service, req, 500, rep, result) || !result || !rep.data())
      std::cerr << "set_pose failed for " << name << "\n";
  }
  return 0;
}
