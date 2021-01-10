#!/usr/bin/env python

import rospy
import cv_bridge
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import math
import numpy as np


#from global marker
import sys
from pylab import *
import numpy as np
import time
from matplotlib import pyplot as plt
import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import *
from std_msgs.msg import *
import tf, geometry_msgs, tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped
#


p = int(input("enter u value :"))
q = int(input("enter v value :"))



test_s=1
test_u=p
test_v=q
# test_R=[[1,0,0],
#         [0,1,0],
#         [0,0,1]]

test_R=[[0,-1,0],
[0,0,-1],
[1,0,0]]


test_t=[[0],[0],[0]]
# test_t=[[0.0175],[0.124226],[-0.042332]]
test_A=[[439.0122919979045 ,0.0, 304.5],
        [0.0,  439.0122919979045,   304.5],
        [0.0  ,0.0,1.0]]


def rotation_matrix(a,b,c):

    M=3
    N=3
    rot_matrix=np.zeroes((3,3))
    rot_matrix[0][0]=np.cos(a)*np.cos(b)
    rot_matrix[0][1]=((np.cos(a)*np.sin(b)*np.sin(c))-(np.sin(a)*np.cos(c)))
    rot_matrix[0][2]=((np.cos(a)*np.sin(b)*np.cos(c))+(np.sin(a)*np.sin(c)))
    rot_matrix[1][0]= np.sin(a)*np.cos(b)
    rot_matrix[1][1]=((np.sin(a)*np.sin(b)*np.sin(c))+(np.cos(a)*np.cos(c)))
    rot_matrix[1][2]=((np.sin(a)*np.sin(b)*np.cos(c))-(np.cos(a)*np.sin(c)))
    rot_matrix[2][0]=np.sin(b)*(-1)
    rot_matrix[2][1]=np.cos(b)*np.sin(c)
    rot_matrix[2][2]=np.cos(b)*np.cos(c)

    return rot_matrix



def main_func(u,v):
    def convert_depth_image(ros_image):
        bridge = CvBridge()
        try :
            depth_image =bridge.imgmsg_to_cv2(ros_image,desired_encoding='passthrough')
        except :
            p=1

        depth_array = np.array(depth_image,dtype=np.dtype(np.float32))
        C=pixel_2_world(test_s,test_u,test_v,test_A,test_R,test_t)
        D=((C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2]))**0.5
        C[0]=C[0]/D
        C[1]=C[1]/D
        C[2]=C[2]/D
        C=C*depth_array[v,u]
        #
        ps = PointStamped()
        ps.header.frame_id = "front_cam_link"
        ps.header.stamp = rospy.Time(0)
        ps.point.x = C[0]
        ps.point.y = C[1]
        ps.point.z = C[2]


        listener = tf.TransformListener()
        mat = listener.transformPoint("base_link", ps)


        ps1 = PointStamped()
        ps1.header.frame_id = "base_link"
        ps1.header.stamp = rospy.Time(0)
        ps1.point.x = mat.point.x
        ps1.point.y = mat.point.y
        ps1.point.z = mat.point.z

        lisn1=tf.TransformListener()
        mat1 = lisn1.transformPoint("/world", ps1)


        print(mat1)
        rospy.loginfo(mat)
        #
        # print(C)
        rospy.loginfo(depth_array)


    def pixel2depth():

        rospy.init_node('pixel2depth',anonymous=True)
        rospy.Subscriber("/front_cam/depth/image_raw",Image,callback=convert_depth_image ,queue_size=1)
        rospy.spin()


    def pixel_2_world(s,u,v,A,R,t):

        k=np.array([[u],[v],[1]])
        k=s*k
        A_inv=np.linalg.inv(A)
        l=np.dot(A_inv,k)
        m=np.subtract(l,t)
        R_inv=np.linalg.inv(R)
        XYZ=np.dot(R_inv,m)
        return XYZ
    pixel2depth()





main_func(p,q)




#1
# test_t=[[0.0175],[0.124226],[-0.042332]]
#2
# test_t=[[0.0175],[0.122466],[-0.041700]]
#3
# test_t=[[0.0175],[0.122466],[-0.05420]]















