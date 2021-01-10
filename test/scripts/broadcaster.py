#!/usr/bin/env python  
import roslib
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
import tf

def handle_pose(msg):
    st = tf2_ros.StaticTransformBroadcaster()
    br = tf.TransformBroadcaster()

    tf2Stamp = TransformStamped()
    tf2Stamp.header.stamp = rospy.Time.now()
    tf2Stamp.header.frame_id = "base_link"
    tf2Stamp.child_frame_id = "front_cam_link"
    tf2Stamp.transform.translation.x = 0.05
    tf2Stamp.transform.translation.y = 0.0
    tf2Stamp.transform.translation.z = -0.06

    quat = tf.transformations.quaternion_from_euler(0.0,0,0.0)

    tf2Stamp.transform.rotation.x = quat[0]
    tf2Stamp.transform.rotation.y = quat[1]
    tf2Stamp.transform.rotation.z = quat[2]
    tf2Stamp.transform.rotation.w = quat[3]

    st.sendTransform(tf2Stamp)

    br.sendTransform((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                     (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w),
                     rospy.Time.now(),
                     "base_link",
                     "/world")
'''    br.sendTransform((0, 0, -0.076),
                     (8.282918624015478e-06, 0.7071792531888889, 8.282918624015478e-06, 0.7070343016586902),
                     rospy.Time.now(),
                     "fpv_link",
                     "base_link") '''





def handle_pose1(msg):
    st1 = tf2_ros.StaticTransformBroadcaster()
    br1 = tf.TransformBroadcaster()

    tf2Stamp1 = TransformStamped()
    tf2Stamp1.header.stamp = rospy.Time.now()
    tf2Stamp1.header.frame_id = "/world"
    tf2Stamp1.child_frame_id = "base_link"
    tf2Stamp1.transform.translation.x = 0.0
    tf2Stamp1.transform.translation.y = 0.0
    tf2Stamp1.transform.translation.z = 0.182464

    quat1 = tf.transformations.quaternion_from_euler(0.0,0,0.0)

    tf2Stamp1.transform.rotation.x = quat1[0]
    tf2Stamp1.transform.rotation.y = quat1[1]
    tf2Stamp1.transform.rotation.z = quat1[2]
    tf2Stamp1.transform.rotation.w = quat1[3]

    st1.sendTransform(tf2Stamp1)

    br1.sendTransform((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                     (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w),
                     rospy.Time.now(),
                     "base_link",
                     "/world")
'''    br.sendTransform((0, 0, -0.076),
                     (8.282918624015478e-06, 0.7071792531888889, 8.282918624015478e-06, 0.7070343016586902),
                     rospy.Time.now(),
                     "fpv_link",
                     "base_link") '''




                     
if __name__ == '__main__':
    rospy.init_node('cam_tf_broadcaster')
    rospy.Subscriber('/ground_truth_to_tf/pose',
                     PoseStamped,
                     handle_pose)
    rospy.Subscriber('/ground_truth_to_tf/pose',
                     PoseStamped,
                     handle_pose1)

    rospy.spin()