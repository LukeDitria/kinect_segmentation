#include <ros/ros.h>
#include <iostream>
#include <math.h> 
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>

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
typedef pcl::Histogram<90> CRH90;
ros::Publisher pub_clusters;
ros::Publisher pub_cloud;
ros::Publisher pub_point;
#define Xoffset 0.846611
#define Yoffset 0.00546124

Eigen::Matrix4f ToPlaneTransform(pcl::ModelCoefficients::Ptr coefficients){
//Returns a Transformation matrix to transform a plane to the XY plane

  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

  float a = coefficients->values[0];
  float b = coefficients->values[1];
  float c = coefficients->values[2];
  float d = coefficients->values[3];

  float cosTheta = (c/sqrt(a*a+b*b+c*c));
  float sinTheta = (sqrt((a*a+b*b)/(a*a+b*b+c*c)));
  float u1 = (b/sqrt(a*a+b*b));
  float u2 = -(a/sqrt(a*a+b*b));

  // Define a rotation matrix
  transform (0,0) = cosTheta + u2 * u2 * (1 - cosTheta);
  transform (0,1) = u1 * u2 * (1 - cosTheta);
  transform (0,2) = u2 * sinTheta;
  transform (1,0) = u1 * u2 * (1 -cosTheta);
  transform (1,1) = cosTheta + u2 * u2 * (1 - cosTheta);
  transform (1,2) = -u1 * sinTheta;
  transform (2,0) = -u2 * sinTheta;
  transform (2,1) = u1 * sinTheta;
  transform (2,2) = cosTheta;
  // subtract z axis component
  transform (0,3) = Yoffset; //y
  transform (1,3) = Xoffset + 0.1; //x 0.917197 +0.05m;
  transform (2,3) = d; //z

  //    (row, column)

return transform;

}

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input)
{
  // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
// Create the segmentation object
pcl::SACSegmentation<pcl::PointXYZ> seg;
pcl::PointCloud<pcl::PointXYZ>::Ptr convexHull(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr objects(new pcl::PointCloud<pcl::PointXYZ>);

visualization_msgs::MarkerArray marker_array;

pcl::fromROSMsg (*input, *cloud);

  //fit a plane to the point cloud 
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);

  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients); 

  Eigen::Matrix4f transform_1  = ToPlaneTransform(coefficients);

  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud1 (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());

  pcl::transformPointCloud (*cloud, *transformed_cloud1, transform_1);
  
  Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
  // theta radians around X axis
  float theta = M_PI/2; // The angle of rotation in radians
  transform_2.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitZ()));
  transform_2.rotate (Eigen::AngleAxisf (M_PI, Eigen::Vector3f::UnitY()));
  pcl::transformPointCloud (*transformed_cloud1, *transformed_cloud, transform_2);


  if (inliers->indices.size() == 0)
		std::cout << "Could not find a plane in the scene." << std::endl;
	else
	{
    //std::cout << transform_1 << std::endl;

     // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    //filtered point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inliers (new pcl::PointCloud<pcl::PointXYZ>);

    extract.setInputCloud (transformed_cloud);
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
		prism.setInputCloud(transformed_cloud);
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

    if (objectIndices->indices.size() == 0){
      		std::cout << "No Objects in the scene." << std::endl;
    }
    else{
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

    int i = 0;
      for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
      {
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

        marker_array.markers[i].header.frame_id = "base_link";
        marker_array.markers[i].header.stamp = ros::Time();
        marker_array.markers[i].ns = "my_namespace";
        marker_array.markers[i].id = i;
        marker_array.markers[i].type = visualization_msgs::Marker::SPHERE;
        marker_array.markers[i].action = visualization_msgs::Marker::ADD;
/*
        marker_array.markers[i].pose.position.x = centroid[2];
        marker_array.markers[i].pose.position.y = -centroid[0];
        marker_array.markers[i].pose.position.z = -centroid[1];
*/
        marker_array.markers[i].pose.position.x = centroid[0];
        marker_array.markers[i].pose.position.y = centroid[1];
        marker_array.markers[i].pose.position.z = centroid[2];

        marker_array.markers[i].pose.orientation.x = 0.0;
        marker_array.markers[i].pose.orientation.y = 0.0;
        marker_array.markers[i].pose.orientation.z = 0.0;
        marker_array.markers[i].pose.orientation.w = 1.0;
        marker_array.markers[i].scale.x = 0.05;
        marker_array.markers[i].scale.y = 0.05;
        marker_array.markers[i].scale.z = 0.05;
        marker_array.markers[i].color.a = 1.0;
        marker_array.markers[i].color.r = 0.0;
        marker_array.markers[i].color.g = 0.1;
        marker_array.markers[i].color.b = 0.1;

        geometry_msgs::Point Centroid_XYZ;
        Centroid_XYZ.x = centroid[0];
        Centroid_XYZ.y = centroid[1];
        Centroid_XYZ.z = centroid[2];
        pub_point.publish (Centroid_XYZ);

        //std::cout << "x: "<< centroid[0] << " y: "<<centroid[1]<< " z: "<<centroid[2] << std::endl;

        i++;
      }
    pub_clusters.publish (marker_array);
  }
}

  pcl::PCLPointCloud2 Cloud_out;

  pcl::toPCLPointCloud2(*transformed_cloud, Cloud_out);
  pub_cloud.publish (Cloud_out);


}

int
main (int argc, char** argv)
{
  // Initialize ROS
  sleep(5);
  ros::init (argc, argv, "kinect_segmentation");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("/camera/depth/points", 1, cloud_cb);

  // Create a ROS publisher for the output model coefficients
  pub_clusters = nh.advertise<visualization_msgs::MarkerArray> ("output", 1);

  pub_cloud = nh.advertise<pcl::PCLPointCloud2> ("output_cloud", 1);

  pub_point = nh.advertise<geometry_msgs::Point> ("output_point", 1);


  // Spin
  ros::spin ();
  return 0; 
}
