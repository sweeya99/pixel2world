# #!/usr/bin/env python
# # license removed for brevity
# import rospy
# #import sensor_msgs.point_cloud2
# #import sensor_msgs.point_cloud2 as pc2

# #from sensor_msgs.msg import PointCloud2
# import sys
# from pylab import *
# import numpy as np
# import time
# from matplotlib import pyplot as plt
# import cv2
# from cv_bridge import CvBridge
# from sensor_msgs.msg import *
# from std_msgs.msg import *
# #import tf, geometry_msgs, tf2_ros
# from geometry_msgs.msg import PointStamped, PoseStamped
# #from tf import TransformBroadcaster




# def callback(val):

#     global x2
#     global y2
    
#     i=0
#     px=x2
#     py=y2
#     print(x2,y2)
#     ''
#     q1,q2,q3,q4 = box
#     a1,b1 = q1
#     a2,b2 = q2
#     a3,b3 = q3
#     a4,b4 = q4
#     for p in pc2.read_points(val, field_names = ("x", "y", "z"), skip_nans=False):
#         if (i==(640*(b1-1))+a1):
#             x1,y1,z1 = p[0],p[1],p[2]
#             break
#         i+=1
#     i=0
#     for p in pc2.read_points(val, field_names = ("x", "y", "z"), skip_nans=False):
#         if (i==(640*(b2-1))+a2):
#             x2,y2,z2 = p[0],p[1],p[2]
#             break
#         i+=1
#     i=0
#     for p in pc2.read_points(val, field_names = ("x", "y", "z"), skip_nans=False):
#         if (i==(640*(b3-1))+a3):
#             x3,y3,z3 = p[0],p[1],p[2]
#             break
#         i+=1
#     i=0
#     for p in pc2.read_points(val, field_names = ("x", "y", "z"), skip_nans=False):
#         if (i==(640*(b4-1))+a4):
#             x4,y4,z4 = p[0],p[1],p[2]
#             break
#         i+=1
#     i=0
#     print(max(z1,z2,z3,z4)-min(z1,z2,z3,z4))
#     fx=(x1+x2+x3+x4)/4
#     fy=(y1+y2+y3+y4)/4
#     fz=(z1+z2+z3+z4)/4
#     print(fx,fy,fz)
 
#     for p in pc2.read_points(val, field_names = ("x", "y", "z"), skip_nans=False):  
#         if (i==(640*(py-1))+px):
#             # print " x : %f  y: %f  z: %f" %(p[0],p[1],p[2])
#             break
#         i+=1
#     print('end of frame',i)

# def cam_frame(data):
#     global x2,y2
#     bridge = CvBridge()
#     frame = bridge.imgmsg_to_cv2(data, "bgr8")

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     l_h =20 # max_value_H*0.1
#     l_s = 70 #max_value*0.3
#     l_v = 70 #max_value*0.35
#     u_h = 50 #max_value_H*0.25
#     u_s = 255
#     u_v = 255

#     lower_red = np.array([l_h, l_s, l_v])
#     upper_red = np.array([u_h, u_s, u_v])

#     mask = cv2.inRange(hsv, lower_red, upper_red)
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.dilate(mask, kernel)

#     # Contours detection
#     if int(cv2.__version__[0]) > 3:
#         # Opencv 4.x.x
#         contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     else:
#         # Opencv 3.x.x
#         _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
#         x = approx.ravel()[0]
#         y = approx.ravel()[1]
#         if area > 200:
#             rect = cv2.minAreaRect(cnt)
#             print(rect)
#             if rect[1][0]>rect[1][1] and rect[2] > -60:
#                 box = cv2.boxPoints(rect)
#                 box = np.int0(box)
#                 im = cv2.drawContours(frame,[box],0,(0,0,255),2)
#                 cv2.circle(frame, (int(rect[0][0]),int(rect[0][1])),5, (255,0,0), 2)
#                 print(box)
#                 p1,p2,p3,p4=box
#                 x1,y1=p1
#                 x2,y2=p3
#                 print(p1,p2,p3,p4)

#                 #cv2.drawContours(frame, [approx], 0, (0, 0, 255), 5)
#                 #cv2.putText(frame, "Rectangle", (int(x), int(y)), font, 1, (0, 100, 0))
#                 """x1=int(rect[0][0]-rect[1][0]/2)
#                 x2=int(rect[0][0]+rect[1][0]/2)
#                 y1=int(rect[0][1]-rect[1][1]/2)
#                 y2=int(rect[0][1]+rect[1][1]/2)"""
#                 cv2.circle(frame, (x1,y1),5, (255,0,0), 2)
#                 cv2.circle(frame, (x1,y2),5, (255,0,0), 2)
#                 cv2.circle(frame, (x2,y1),5, (0,255,0), 2)
#                 cv2.circle(frame, (x2,y2),5, (0,255,0), 2)


#     cv2.imshow("Mask", mask)
#     cv2.imshow("Frame", frame)
#     cv2.waitKey(30)

# '''
# def callback():
#     rospy.loginfo(rospy.get_caller_id())
#     x = np.random.random()
#     x1 = 3.9 + (0.2*x)
#     x = np.random.random()
#     x2 = 5.4 + (0.2*x)
#     x = np.random.random()
#     x3 = 6.9 + (0.2*x)
#     x = np.random.random()
#     x4 = 8.4 + (0.2*x)
#     x = np.random.random()
#     x5 = 9.9 + (0.2*x)
#     x = np.random.random()
#     y1 = -0.1 + (0.2*x)
#     x = np.random.random()
#     y2 = -1.6 + (0.2*x)
#     x = np.random.random()
#     y3 = 1.4 + (0.2*x)
#     x = np.random.random()
#     y4 = -0.11 + (0.2*x)
#     x = np.random.random()
#     y5 = 0.9 + (0.2*x)
#     x = np.random.random()
#     z1 = 3.27 + (0.05*x)
#     x = np.random.random()
#     z2 = 3.27 + (0.05*x)
#     x = np.random.random()
#     z3 = 3.27 + (0.05*x)
#     x = np.random.random()
#     z4 = 3.27 + (0.05*x)
#     x = np.random.random()
#     z5 = 3.27 + (0.05*x)

#     print " Frame 1- x : %f  y: %f  z: %f" %(x1,y1,z1)
#     print " Frame 2- x : %f  y: %f  z: %f" %(x2,y2,z2)
#     print " Frame 3- x : %f  y: %f  z: %f" %(x3,y3,z3)
#     print " Frame 4- x : %f  y: %f  z: %f" %(x4,y4,z4)
#     print " Frame 5- x : %f  y: %f  z: %f" %(x5,y5,z5)

# '''

# if __name__ == '__main__':
#     rospy.init_node('world_coordinate', anonymous=True)
#     rospy.Subscriber("/front_cam/color/image_raw", Image, cam_frame)
#     rospy.Subscriber("/front_cam/depth/points", PointCloud2, callback)
#     # rospy.Subscriber("/mybot/depth_camera/depth/image_raw", Image, cam_frame)
#     # rospy.Subscriber("/mybot/depth_camera/depth_points_topic", PointCloud2, callback)
#     rospy.spin()

# # /front_cam/color/image_raw

# #     /front_cam/depth/points



#!/usr/bin/env python

import rospy
import cv_bridge 
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
import math


R=[[0 ,0,1], 
        [-1,0 , 0],
        [0 , -1, 0] ]



R_inv=np.linalg.inv(R)

print(R_inv)