// Author: Frank Loewenich
// Persistent Gazebo client for the replay renderer. Reads commands from stdin:
//   <model> x y z qx qy qz qw   set a model's world pose (blocks until the request is answered)
//   step N                      step the paused world by N iterations (blocks until answered)
//   ping                        print "ok" (flush point for lockstep)
// Used by scripts/replay_follow.py so the camera renders exactly one frame per set pose.
//   gz_set_pose <world>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/world_control.pb.h>
#include <gz/transport/Node.hh>

#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char **argv) {
  if (argc < 2) { std::cerr << "usage: gz_set_pose <world>\n"; return 1; }
  const std::string world = argv[1];
  const std::string pose_service = "/world/" + world + "/set_pose";
  const std::string control_service = "/world/" + world + "/control";
  gz::transport::Node node;
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string head;
    if (!(in >> head)) continue;
    if (head == "ping") { std::cout << "ok" << std::endl; continue; }
    if (head == "step") {
      unsigned n = 1; in >> n;
      gz::msgs::WorldControl req;
      req.set_multi_step(n);
      req.set_pause(true);
      gz::msgs::Boolean rep; bool result = false;
      if (!node.Request(control_service, req, 2000, rep, result) || !result || !rep.data())
        std::cerr << "step failed\n";
      std::cout << "stepped" << std::endl;
      continue;
    }
    double x, y, z, qx, qy, qz, qw;
    if (!(in >> x >> y >> z >> qx >> qy >> qz >> qw)) continue;
    gz::msgs::Pose req;
    req.set_name(head);
    req.mutable_position()->set_x(x); req.mutable_position()->set_y(y); req.mutable_position()->set_z(z);
    req.mutable_orientation()->set_x(qx); req.mutable_orientation()->set_y(qy);
    req.mutable_orientation()->set_z(qz); req.mutable_orientation()->set_w(qw);
    gz::msgs::Boolean rep; bool result = false;
    if (!node.Request(pose_service, req, 2000, rep, result) || !result || !rep.data())
      std::cerr << "set_pose failed for " << head << "\n";
  }
  return 0;
}
