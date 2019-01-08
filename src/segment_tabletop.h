//System
#include <iostream>
#include <math.h>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm> 
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 

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

//GLOBAL VARIABLES
float cylinder_offset = 0.025;

class SegmentTabletop {

 private:
  ros::NodeHandle nh_;
  ros::Subscriber point_cloud_sub_;
  ros::Publisher object_markers_pub_;
  tf::TransformListener transformer;
  tf::TransformListener listener;
  std::string point_cloud_topic;
  std::string out_object_markers_topic;
  visualization_msgs::MarkerArray marker_array;
  actionlib::SimpleActionServer<kinect_segmentation::ScanObjectsAction> as_;
  sensor_msgs::PointCloud2 input_cloud;
  std::mutex cloud_mutex;
  double filter_base_link_radius;
  
  void init_params(){    
    nh_.getParam("point_cloud_topic", point_cloud_topic);
    nh_.getParam("out_object_markers_topic", out_object_markers_topic);
    nh_.getParam("filter_base_link_radius", filter_base_link_radius);
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
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr convexHull(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr objects(new pcl::PointCloud<pcl::PointXYZRGB>);

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
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    // Filtered point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_inliers (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*cloud_inliers);

    // Retrieve the convex hull.
    pcl::ConvexHull<pcl::PointXYZRGB> hull;
    hull.setInputCloud(cloud_inliers);

    // Make sure that the resulting hull is bidimensional.
    hull.setDimension(2);
    hull.reconstruct(*convexHull);

    // Prism object from convex hull
    pcl::ExtractPolygonalPrismData<pcl::PointXYZRGB> prism;
    prism.setInputCloud(cloud);
    prism.setInputPlanarHull(convexHull);
    
    // First parameter: minimum Z value. Set to 0, segments objects lying on the plane (can be negative).
    // Second parameter: maximum Z value, set to 10cm. Tune it according to the height of the objects you expect.
    prism.setHeightLimits(0.02f, 0.1f); // Min then Max
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
    
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud (objects);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance (0.02); // 2cm
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (15000); //25000
    ec.setSearchMethod (tree);
    ec.setInputCloud (objects);
    ec.extract (cluster_indices);

    marker_array.markers.resize(cluster_indices.size());
    //std::cout<<cluster_indices.size() << " Objects Found.."<<std::endl;
    
    
    int i = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
      float avg_r = 0, avg_g = 0, avg_b = 0, counter = 0;
      

      for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
        // Summing up rgb values of points in object_cluster point clouds
        avg_r += objects->points[*pit].r;
        avg_g += objects->points[*pit].g;
        avg_b += objects->points[*pit].b;  

        object_cluster->points.push_back (objects->points[*pit]); //*
        
        counter++;
      }

      // Finding average rgb values of cylinders and avg z value
      avg_r = avg_r/counter, avg_g = avg_g/counter, avg_b = avg_b/counter;
      
      object_cluster->width = object_cluster->points.size ();
      object_cluster->height = 1;
      object_cluster->is_dense = true;

      
      pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normalEstimation;
      pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
      normalEstimation.setInputCloud(object_cluster);
      normalEstimation.setRadiusSearch(0.03);

      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
      normalEstimation.setSearchMethod(kdtree);
      normalEstimation.compute(*normals);

      // CRH estimation object.
      pcl::CRHEstimation<pcl::PointXYZRGB, pcl::Normal, CRH90> crh;
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
      //transform point to base_link frame
      listener.transformPoint("base_link",Centroid_Camera_Link,Centroid_Base_Link);



      float X_pos = Centroid_Base_Link.point.x;
      float Y_pos = Centroid_Base_Link.point.y;
      float Z_pos = Centroid_Base_Link.point.z;
      //filter out anything that could be the base of the robot or the gripper
      if (sqrt(X_pos*X_pos+Y_pos*Y_pos+Z_pos*Z_pos) > filter_base_link_radius ){
        
        result_.centroids.push_back(Centroid_Base_Link);

        marker_array.markers[i].header.frame_id = Centroid_Base_Link.header.frame_id;
        marker_array.markers[i].header.stamp = ros::Time();
        //marker_array.markers[i].ns = "my_namespace";
        marker_array.markers[i].id = i;
        marker_array.markers[i].type = visualization_msgs::Marker::SPHERE;
        marker_array.markers[i].action = visualization_msgs::Marker::ADD;
        
        // Positioning markers 
        float x_offset = X_pos * (cylinder_offset/sqrt(X_pos*X_pos+Y_pos*Y_pos));
        float y_offset = Y_pos * (cylinder_offset/sqrt(X_pos*X_pos+Y_pos*Y_pos));
        marker_array.markers[i].pose.position.x = X_pos - x_offset;
        if(X_pos > 0){
        	marker_array.markers[i].pose.position.y = Y_pos - y_offset;
        }
        else{
        	marker_array.markers[i].pose.position.y = Y_pos + y_offset;
        }
		
        marker_array.markers[i].pose.orientation.x = 0.0;
        marker_array.markers[i].pose.orientation.y = 0.0;
        marker_array.markers[i].pose.orientation.z = 0.0;
        marker_array.markers[i].pose.orientation.w = 1.0;
        marker_array.markers[i].scale.x = 0.05;
        marker_array.markers[i].scale.y = 0.05;
        marker_array.markers[i].scale.z = 0.05;
        marker_array.markers[i].color.a = 1.0;
        //ROS_INFO("This is centroid number %.4f", i);
        
        // Identifying cylinder colour
        marker_array.markers[i].color.g = 0.0;
        if(avg_r < avg_b){
          marker_array.markers[i].color.r = 1.0;
          marker_array.markers[i].color.b = 0.0;
        }
        else{
          marker_array.markers[i].color.r = 0.0;
          marker_array.markers[i].color.b = 1.0;
        }

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