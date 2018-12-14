//System
#include <iostream>
#include <math.h>
#include <mutex>
#include <thread>

//ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <actionlib/server/simple_action_server.h>

//PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/centroid.h>
#include <pcl/features/crh.h>
#include <pcl/common/transforms.h>

//CUSTOM
#include <kinect_segmentation/ScanObjectsAction.h>
typedef pcl::Histogram<90> CRH90;

class SegmentTabletop {

 private:
  ros::NodeHandle nh_;
  ros::Subscriber point_cloud_sub_;
  ros::Publisher object_markers_pub_;
  tf::TransformListener listener;
  std::string point_cloud_topic;
  std::string out_object_markers_topic;
  visualization_msgs::MarkerArray marker_array;
  actionlib::SimpleActionServer<kinect_segmentation::ScanObjectsAction> as_;
  sensor_msgs::PointCloud2 input_cloud;
  std::mutex cloud_mutex;
  
  void init_params(){    
    nh_.getParam("point_cloud_topic", point_cloud_topic);
    nh_.getParam("out_object_markers_topic", out_object_markers_topic);
  }  
  void init_subs(){
    point_cloud_sub_ = nh_.subscribe(point_cloud_topic, 1, &SegmentTabletop::cloudCB, this); 
  }
  void init_pubs(){
    object_markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray> (out_object_markers_topic, 1);
  }
  void init_actionlib(){
    as_.start();
    ROS_INFO("Scan Objects Server ON");
  }      
  void cloudCB (const sensor_msgs::PointCloud2ConstPtr& input) {
    std::lock_guard<std::mutex> lock{cloud_mutex};
    input_cloud = *input;
  }
  void executeCB(const actionlib::SimpleActionServer<kinect_segmentation::ScanObjectsAction>::GoalConstPtr& goal){
    std::lock_guard<std::mutex> lock{cloud_mutex};
    ROS_INFO("executeCB: ScanObjects");        
    
    kinect_segmentation::ScanObjectsResult result_;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr convexHull(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr objects(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::fromROSMsg (input_cloud, *cloud);

    //fit a plane to the point cloud
    seg.setOptimizeCoefficients (true); // Optional
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);
  
    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients); 

    if (inliers->indices.size() == 0) {      
      std::cout << "Could not find a plane in the scene." << std::endl;
      as_.setAborted(result_);
      return;
    }

    //std::cout << "Plane Found" << std::endl;

    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    //filtered point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inliers (new pcl::PointCloud<pcl::PointXYZ>);
    
    extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*cloud_inliers);

    // Retrieve the convex hull.
    pcl::ConvexHull<pcl::PointXYZ> hull;
    hull.setInputCloud(cloud_inliers);

    // Make sure that the resulting hull is bidimensional.
    hull.setDimension(2);
    hull.reconstruct(*convexHull);

    // Prism object from convex hull
    pcl::ExtractPolygonalPrismData<pcl::PointXYZ> prism;
    prism.setInputCloud(cloud);
    prism.setInputPlanarHull(convexHull);
    
    // First parameter: minimum Z value. Set to 0, segments objects lying on the plane (can be negative).
    // Second parameter: maximum Z value, set to 10cm. Tune it according to the height of the objects you expect.
    prism.setHeightLimits(0.02f, 0.5f); // Min then Max
    //prism.setHeightLimits(-0.5f,-0.02f); // Min then Max
    pcl::PointIndices::Ptr objectIndices(new pcl::PointIndices);

    prism.segment(*objectIndices);

    // Get and show all points retrieved by the hull.
    extract.setIndices(objectIndices);
    extract.filter(*objects);
    
    if (objectIndices->indices.size() == 0) {
      //std::cout << "No Objects in the scene." << std::endl;
      as_.setAborted(result_);
      return;
    }
    
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (objects);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.02); // 2cm
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (objects);
    ec.extract (cluster_indices);

    marker_array.markers.resize(cluster_indices.size());
    //std::cout<<cluster_indices.size() << " Objects Found.."<<std::endl;


    int i = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr object_cluster (new pcl::PointCloud<pcl::PointXYZ>);
      for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
	object_cluster->points.push_back (objects->points[*pit]); //*
      }
      object_cluster->width = object_cluster->points.size ();
      object_cluster->height = 1;
      object_cluster->is_dense = true;

      pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
      pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
      normalEstimation.setInputCloud(object_cluster);
      normalEstimation.setRadiusSearch(0.03);

      pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
      normalEstimation.setSearchMethod(kdtree);
      normalEstimation.compute(*normals);

      // CRH estimation object.
      pcl::CRHEstimation<pcl::PointXYZ, pcl::Normal, CRH90> crh;
      crh.setInputCloud(object_cluster);
      crh.setInputNormals(normals);
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*object_cluster, centroid);
      crh.setCentroid(centroid);

      geometry_msgs::PointStamped Centroid_Camera_Link;
      geometry_msgs::PointStamped Centroid_Base_Link;

      Centroid_Camera_Link.header.frame_id = input_cloud.header.frame_id;
      Centroid_Camera_Link.point.x = centroid[0];
      Centroid_Camera_Link.point.y = centroid[1];
      Centroid_Camera_Link.point.z = centroid[2];
      //transform point o base_link frame
      listener.transformPoint("base_link",Centroid_Camera_Link,Centroid_Base_Link);

      float X_pos = Centroid_Camera_Link.point.x;
      float Y_pos = Centroid_Camera_Link.point.y;
      float Z_pos = Centroid_Camera_Link.point.z;
      //filter out anything that could be the base of the robot
      if (sqrt(X_pos*X_pos+Y_pos*Y_pos+Z_pos*Z_pos) > 0.3) {
        
        result_.centroids.push_back(Centroid_Base_Link);

        marker_array.markers[i].header.frame_id = Centroid_Base_Link.header.frame_id;
        marker_array.markers[i].header.stamp = ros::Time();
        marker_array.markers[i].ns = "my_namespace";
        marker_array.markers[i].id = i;
        marker_array.markers[i].type = visualization_msgs::Marker::SPHERE;
        marker_array.markers[i].action = visualization_msgs::Marker::ADD;
        marker_array.markers[i].pose.position.x = X_pos;
        marker_array.markers[i].pose.position.y = Y_pos;
        marker_array.markers[i].pose.position.z = Z_pos;
        marker_array.markers[i].pose.orientation.x = 0.0;
        marker_array.markers[i].pose.orientation.y = 0.0;
        marker_array.markers[i].pose.orientation.z = 0.0;
        marker_array.markers[i].pose.orientation.w = 1.0;
        marker_array.markers[i].scale.x = 0.05;
        marker_array.markers[i].scale.y = 0.05;
        marker_array.markers[i].scale.z = 0.05;
        marker_array.markers[i].color.a = 1.0;
        marker_array.markers[i].color.r = 0.0;
        marker_array.markers[i].color.g = 0.9;
        marker_array.markers[i].color.b = 0.2;

        kinect_segmentation::ScanObjectsAction action;
        i++;          
      }  
    }
    object_markers_pub_.publish (marker_array);    
    as_.setSucceeded(result_);
  }
  
public:

 SegmentTabletop(ros::NodeHandle* nodehandle): nh_(*nodehandle), as_(nh_, "/scan_objects", boost::bind(&SegmentTabletop::executeCB, this, _1),false) {
    init_params();
    init_subs();
    init_pubs();
    init_actionlib();
  }
  
};
