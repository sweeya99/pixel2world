#!/usr/bin/env python

import rospy
import cv_bridge 
from cv_bridge import CvBridge
#import CvBridge#, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
i=2
def convert_depth_image(ros_image):
    print("1")
    bridge = CvBridge()
     # Use cv_bridge() to convert the ROS image to OpenCV format
    try :
        print(i)
        print("1")
     #Convert the depth image using the default passthrough encoding
     	depth_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding='passthrough')
        print("3")
    except :
     	p=1
 	#  	# print e

    #Convert the depth image to a Numpy array
    print("4")
    depth_array = np.array(depth_image,dtype =np.dtype(np.float32) )
    print(depth_array[314,304])

    print(depth_array.shape)

     

    rospy.loginfo(depth_array)



def pixel2depth():
    
	rospy.init_node('pixel2depth',anonymous=True)
	rospy.Subscriber("/mybot/depth_camera/depth/image_raw", Image,callback=convert_depth_image, queue_size=1)
	rospy.spin()
    
    

if __name__ == '__main__':
	pixel2depth()


