# #/usr/bin/env python

# import rospy
# import numpy as np
# import cv2
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
# cvb=CvBridge()
# i=0
# depth_mem=None
# first_flag=True


# def imnormalize(xmax,image):
#     """
#     Normalize a list of sample image data in the range of 0 to 1
#     : image:image data.
#     : return: Numpy array of normalize data
#     """
#     xmin = 0
#     a = 0
#     b = 255
    
#     return ((np.array(image,dtype=np.float32) - xmin) * (b - a)) / (xmax - xmin)

# def checkdepth(msg):
#     global i
#     global video
#     global depth_mem
#     global first_flag
#     try:
#         cv_image = cvb.imgmsg_to_cv2(msg,msg.encoding)
#     except CvBridgeError as e:
#         print(e)
    
#     image_normal= np.array(imnormalize(np.max(cv_image),cv_image),dtype=np.uint8)
#     numpy_image= np.array(cv_image,dtype=np.uint16)
#     if first_flag == True:
#         depth_mem = np.copy(numpy_image)
#         np.save("./dframe"+str(i)+".npy",numpy_image)
#         cv2.imwrite("./dframe"+str(i)+".jpg", image_normal)
#         first_flag=False
#     if (depth_mem==numpy_image).all() :
#         return
#     else:
#         depth_mem = np.copy(numpy_image)
#         np.save("./dframe"+str(i)+".npy",numpy_image)
#         cv2.imwrite("./dframe"+str(i)+".jpg", image_normal)
#     i+=1
    
# if __name__ == '__main__':
    
#     rospy.init_node("depthtest")
    
#     rate=rospy.Rate(1) #25hz
#     rospy.loginfo("Running depth Grabber")
#     while not rospy.is_shutdown():
#         rospy.Subscriber("/mybot/depth_camera/color/image_raw",Image,checkdepth)
    
#         rate.sleep()

#!/usr/bin/env python
import rospy
import cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from numpy as np

def convert_depth_image(self, ros_image):
     bridge = CvBridge()
     # Use cv_bridge() to convert the ROS image to OpenCV format
      try:
     #Convert the depth image using the default passthrough encoding
                depth_image = cv_bridge.imgmsg_to_cv2(ros_image, deside_encoding="passthrough")

      except CvBridgeError, e:
 	          print e
     #Convert the depth image to a Numpy array
      depth_array = np.array(depth_image, dtype=np.float32)

      rospy.loginfo(depth_array)

def pixel2depth():
	rospy.init_node('pixel2depth',anonymous=True)
	rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image,callback=convert_depth_image, queue_size=1)
	rospy.spin()

if __name__ == '__main__':
	pixel2depth()