//System
#include <iostream>
#include <string>
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

class SegmentTabletop {

 private:
  ros::NodeHandle nh_;
  ros::Subscriber point_cloud_sub_;
  ros::Publisher object_markers_pub_;

  ros::Publisher goal_markers_pub_;
  tf::TransformListener listener;
  std::string point_cloud_topic;
  std::string out_object_markers_topic;

  std::string out_goal_markers_topic;
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::MarkerArray goals_array;
  actionlib::SimpleActionServer<kinect_segmentation::ScanObjectsAction> as_;
  sensor_msgs::PointCloud2 input_cloud;
  std::mutex cloud_mutex;
  double filter_base_link_radius;
  double redGoal_x, redGoal_y, redGoal_z, blueGoal_x, blueGoal_y, blueGoal_z;
  bool simulation;

  void init_params(){    
    nh_.getParam("point_cloud_topic", point_cloud_topic);
    nh_.getParam("out_object_markers_topic", out_object_markers_topic);
    nh_.getParam("filter_base_link_radius", filter_base_link_radius);
    nh_.getParam("out_goal_markers_topic", out_goal_markers_topic);
    nh_.getParam("redGoal_x", redGoal_x);
    nh_.getParam("redGoal_y", redGoal_y);
    nh_.getParam("redGoal_z", redGoal_z);
    nh_.getParam("blueGoal_x", blueGoal_x);
    nh_.getParam("blueGoal_y", blueGoal_y);
    nh_.getParam("blueGoal_z", blueGoal_z);
    nh_.getParam("simulation", simulation);

  }  
  void init_subs(){
    point_cloud_sub_ = nh_.subscribe(point_cloud_topic, 1, &SegmentTabletop::cloudCB, this); 
  }
  void init_pubs(){
    object_markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray> (out_object_markers_topic, 1);
    goal_markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray> (out_goal_markers_topic, 1);
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
    goals_array.markers.resize(2);
    //std::cout<<cluster_indices.size() << " Objects Found.."<<std::endl;
    
    int i = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
      float avg_r = 0, avg_g = 0, avg_b = 0, counter = 0, z_avg = 0;

      geometry_msgs::PointStamped base_temp;
      geometry_msgs::PointStamped camera_temp;
      std::vector<float> x_vals;
      std::vector<float> y_vals;

      for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
        
        // Summing up rgb values of points in object_cluster point clouds
        avg_r += objects->points[*pit].r;
        avg_g += objects->points[*pit].g;
        avg_b += objects->points[*pit].b; 

        counter++;

        object_cluster->points.push_back (objects->points[*pit]); 
        
        camera_temp.header.frame_id = input_cloud.header.frame_id;
        camera_temp.point.x = objects->points[*pit].x;
        camera_temp.point.y = objects->points[*pit].y;
        camera_temp.point.z = objects->points[*pit].z;

        //transform point to base_link frame
        listener.transformPoint("base_link",camera_temp,base_temp);

        x_vals.push_back(base_temp.point.x); 
        y_vals.push_back(base_temp.point.y);
        z_avg += base_temp.point.z; 
      }

      // Finding average rgb values and z value of cylinder
      avg_r = avg_r/counter, avg_g = avg_g/counter, avg_b = avg_b/counter, z_avg = z_avg/counter;

      // Fitting ellipse to cirular top of cylinder
      Eigen::MatrixXd D_mat (x_vals.size(),6);
      Eigen::MatrixXd D_mat_trans (6, x_vals.size());
      Eigen::MatrixXd S_mat (6, 6);
      Eigen::MatrixXd C_mat = Eigen::MatrixXd::Constant(6, 6, 0);

      // D = [x.*x, x.*y, y.*y, x, y, ones];
      for(int i=0; i<x_vals.size(); i++){
        D_mat(i,0) = x_vals[i]*x_vals[i];
        D_mat(i,1) = x_vals[i]*y_vals[i];
        D_mat(i,2) = y_vals[i]*y_vals[i];
        D_mat(i,3) = x_vals[i];
        D_mat(i,4) = y_vals[i];
        D_mat(i,5) = 1.0;
      }
      
      D_mat_trans = D_mat.transpose();
      S_mat = D_mat_trans * D_mat;

      // Build 6x6 constraint matrix
      C_mat(5,5) = 0; C_mat(0,2) = 2; C_mat(1,1) = -1; C_mat(2,0) = 2;

      // Solving eigensystem
      Eigen::EigenSolver< Eigen::MatrixXd > es(S_mat.inverse() * C_mat);
      // Finding positive eigenvalue
      int posEigen;
      for(int i=0; i<es.eigenvalues().size(); i++){
        if(es.eigenvalues().real()[i]>0.01){
          posEigen = i;
          //std::cout << "\nEigenvalue: ";
        //std::cout << es.eigenvalues().real()[i];
        //std::cout << "\nEigenvectors: ";
        //std::cout << es.eigenvectors().real().col(i);
        }
      }
      
      // Finding the centre of the fitted ellipse
      float a,b,c,d,e,x0,y0,R = 0;
      a = es.eigenvectors().real().col(posEigen)[0];
      b = es.eigenvectors().real().col(posEigen)[1]/2.0;
      c = es.eigenvectors().real().col(posEigen)[2];
      d = es.eigenvectors().real().col(posEigen)[3]/2.0;
      e = es.eigenvectors().real().col(posEigen)[4]/2.0;
      
      x0=(c*d-b*e)/(b*b-a*c);
      y0=(a*e-b*d)/(b*b-a*c);

      //Finding radius of cylinder
      for(int i=0; i<x_vals.size(); i++){
        R += (x0 - x_vals[i])*(x0 - x_vals[i]) + (y0 - y_vals[i])*(y0 - y_vals[i]);
      }
      R = R/x_vals.size();
      R = sqrt(R);

    
      object_cluster->width = object_cluster->points.size ();
      object_cluster->height = 1;
      object_cluster->is_dense = true;

      /*
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
    */


      geometry_msgs::PointStamped Centroid_Camera_Link;
      geometry_msgs::PointStamped Centroid_Base_Link;

      Centroid_Camera_Link.header.frame_id = input_cloud.header.frame_id;

      //transform points to base_link frame
      listener.transformPoint("base_link",Centroid_Camera_Link,Centroid_Base_Link);


      Centroid_Base_Link.point.x = x0;
      Centroid_Base_Link.point.y = y0;
      Centroid_Base_Link.point.z = z_avg;

      float X_pos = Centroid_Base_Link.point.x;
      float Y_pos = Centroid_Base_Link.point.y;
      float Z_pos = Centroid_Base_Link.point.z;
      // filter out anything that could be the base of the robot or the gripper or behind the gripper
      if (sqrt(X_pos*X_pos+Y_pos*Y_pos+Z_pos*Z_pos) > filter_base_link_radius && X_pos < 0)
    	{
    	  result_.centroids.push_back(Centroid_Base_Link);
	  result_.radiuses.push_back(R);
    	  
    	  marker_array.markers[i].header.frame_id = Centroid_Base_Link.header.frame_id;
    	  marker_array.markers[i].header.stamp = ros::Time();
    	  //marker_array.markers[i].ns = "my_namespace";
    	  marker_array.markers[i].id = i;
    	  marker_array.markers[i].type = visualization_msgs::Marker::CYLINDER;
    	  marker_array.markers[i].action = visualization_msgs::Marker::ADD;
            

    	  /*
    	  // Positioning markers - linear method (not super accurate)
    	  float cylinder_offset = 0.025;
    	  float x_offset = X_pos * (cylinder_offset/sqrt(X_pos*X_pos+Y_pos*Y_pos));
    	  float y_offset = Y_pos * (cylinder_offset/sqrt(X_pos*X_pos+Y_pos*Y_pos));
    	  marker_array.markers[i].pose.position.x = X_pos - x_offset;
    	  if(X_pos > 0){
              marker_array.markers[i].pose.position.y = Y_pos - y_offset;
    	  }
    	  else{
              marker_array.markers[i].pose.position.y = Y_pos + y_offset;
    	  }
    	  */
            
    	  marker_array.markers[i].pose.position.x = X_pos;
    	  marker_array.markers[i].pose.position.y = Y_pos;
    	  marker_array.markers[i].pose.position.z = Z_pos;
    	  marker_array.markers[i].pose.orientation.x = 0.0;
    	  marker_array.markers[i].pose.orientation.y = 0.0;
    	  marker_array.markers[i].pose.orientation.z = 0.0;
    	  marker_array.markers[i].pose.orientation.w = 1.0;
    	  marker_array.markers[i].scale.x = 0.06;//2*R;
    	  marker_array.markers[i].scale.y = 0.06;//2*R;
    	  marker_array.markers[i].scale.z = 0.12;//4*Z_pos;
    	  marker_array.markers[i].color.a = 1.0;
    	  //ROS_INFO("This is centroid number %.4f", i);

    	  std::string color;
    	  // Identifying cylinder colour
    	  marker_array.markers[i].color.g = 0.0;
    	  
    	  if(simulation){
    	    if(avg_r < avg_b){
    	      marker_array.markers[i].color.r = 1.0;
    	      marker_array.markers[i].color.b = 0.0;
    	      color = "red";
    	    }
    	    else{
    	      marker_array.markers[i].color.r = 0.0;
    	      marker_array.markers[i].color.b = 1.0;
    	      color = "blue";
    	    }
    	  }
    	  else{
    	    if(avg_r > avg_b){
    	      marker_array.markers[i].color.r = 1.0;
    	      marker_array.markers[i].color.b = 0.0;
    	      color = "red";
    	    }
    	    else{
    	      marker_array.markers[i].color.r = 0.0;
    	      marker_array.markers[i].color.b = 1.0;
    	      color = "blue";
    	    }
    	  }
    	  result_.colors.push_back(color);

    	  i++;          
    	}  

    }

    int j = 0;
    
        
    // Goal Regions -
    geometry_msgs::PointStamped redGoal;
    geometry_msgs::PointStamped blueGoal;

    redGoal.point.x = redGoal_x;
    redGoal.point.y = redGoal_y;
    redGoal.point.z = redGoal_z;
    blueGoal.point.x = blueGoal_x;
    blueGoal.point.y = blueGoal_y;
    blueGoal.point.z = blueGoal_z;
    
    
    geometry_msgs::PointStamped redGoal_base_link;
    geometry_msgs::PointStamped blueGoal_base_link;

    redGoal.header.frame_id = input_cloud.header.frame_id;
    blueGoal.header.frame_id = input_cloud.header.frame_id;

    listener.transformPoint("base_link",redGoal,redGoal_base_link);
    listener.transformPoint("base_link",blueGoal,blueGoal_base_link);

    redGoal_base_link.point.z = 0;
    blueGoal_base_link.point.z = 0;
    
    result_.red_goal = redGoal_base_link;
    result_.blue_goal = blueGoal_base_link; 

    // Red Goal
    goals_array.markers[j].header.frame_id = redGoal_base_link.header.frame_id;
    goals_array.markers[j].header.stamp = ros::Time();
    //goals_array.markers[j].ns = "my_namespace";
    goals_array.markers[j].id = j;
    goals_array.markers[j].type = visualization_msgs::Marker::CYLINDER;
    goals_array.markers[j].action = visualization_msgs::Marker::ADD; 
    goals_array.markers[j].pose.position.x = redGoal_base_link.point.x;
    goals_array.markers[j].pose.position.y = redGoal_base_link.point.y;
    goals_array.markers[j].pose.position.z = redGoal_base_link.point.z;
    goals_array.markers[j].pose.orientation.x = 0.0;
    goals_array.markers[j].pose.orientation.y = 0.0;
    goals_array.markers[j].pose.orientation.z = 0.0;
    goals_array.markers[j].pose.orientation.w = 1.0;
    goals_array.markers[j].scale.x = 0.4;//2*R;
    goals_array.markers[j].scale.y = 0.4;//2*R;
    goals_array.markers[j].scale.z = 0.005;//4*Z_pos;
    goals_array.markers[j].color.a = 0.7;
    goals_array.markers[j].color.r = 1.0;
    goals_array.markers[j].color.g = 0.0;
    goals_array.markers[j].color.b = 0.0;
    j++;
     

    // Blue Goal
    goals_array.markers[j].header.frame_id = blueGoal_base_link.header.frame_id;
    goals_array.markers[j].header.stamp = ros::Time();
    //goals_array.markers[j].ns = "my_namespace";
    goals_array.markers[j].id = j;
    goals_array.markers[j].type = visualization_msgs::Marker::CYLINDER;
    goals_array.markers[j].action = visualization_msgs::Marker::ADD;
    goals_array.markers[j].pose.position.x = blueGoal_base_link.point.x;
    goals_array.markers[j].pose.position.y = blueGoal_base_link.point.y;
    goals_array.markers[j].pose.position.z = blueGoal_base_link.point.z;
    goals_array.markers[j].pose.orientation.x = 0.0;
    goals_array.markers[j].pose.orientation.y = 0.0;
    goals_array.markers[j].pose.orientation.z = 0.0;
    goals_array.markers[j].pose.orientation.w = 1.0;
    goals_array.markers[j].scale.x = 0.4;//2*R;
    goals_array.markers[j].scale.y = 0.4;//2*R;
    goals_array.markers[j].scale.z = 0.005;//4*Z_pos;
    goals_array.markers[j].color.a = 0.7;
    goals_array.markers[j].color.r = 0.0;
    goals_array.markers[j].color.g = 0.0;
    goals_array.markers[j].color.b = 1.0;
	   
	   

    object_markers_pub_.publish (marker_array);    
    goal_markers_pub_.publish (goals_array); 
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
