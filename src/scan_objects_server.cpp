#include "segment_tabletop.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "scan_objects_server");
  ros::NodeHandle node_handle("~");
  SegmentTabletop segment_tabletop(&node_handle);
  ros::spin();
  return 0;
}
