#!/usr/bin/env python

import rospy
import cv_bridge 
from cv_bridge import CvBridge
#import CvBridge#, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
import numpy as np
#import cv2
test_s=1
test_u=262
test_v=152

# quad wrt world r,t, matrix
test_R=[[1 ,0,0],
        [0 ,1 , 0],
        [0 , 0 , 1] ]



#test_r2 is fixed and is camera wrt quad rotation matrix

# test_R_2=[[0.9238795325 ,0,0.3826834324],
#         [0 ,1 , 0],
#         [-0.3826834324 , 0 ,0.9238795325] ]

# test_R=np.dot(test_R_1,test_R_2)

# print(test_R)

test_t=[[0.0], [0.0],[0.0]]
# test_t=[[-0.037668], [0.018656],[0.11624]]
test_A=[[476.7030836014194 ,0.0,400.5],
        [0.0,  476.7030836014194,   400.5],
        [0.0  ,0.0,1.0]]



def main_func(u,v):
    
    

	def convert_depth_image(ros_image):
	    # print("1")
	    bridge = CvBridge()
	     # Use cv_bridge() to convert the ROS image to OpenCV format
	    try :
	        # print(i)
	        # print("1")
	     #Convert the depth image using the default passthrough encoding
	     	depth_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding='passthrough')
	        # print("3")
	    except :
	     	p=1
	 	#  	# print e

	    #Convert the depth image to a Numpy array
	    # print("4")
	    depth_array = np.array(depth_image,dtype =np.dtype(np.float32) )
	    # a=depth_array[v,u]
	    #depth_array[u,v]
	    C=pixel_2_world(test_s,test_u,test_v,test_A,test_R,test_t)
	    # print(depth_array.shape)
	    D=((C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2]))**0.5

	    # GETTING THE UNIT VECTOR
	    C[0]=C[0]/D
	    C[1]=C[1]/D
	    C[2]=C[2]/D

	    C=C*depth_array[u,v]

	    print(C)

	     

	    rospy.loginfo(depth_array)



	def pixel2depth():
	    
		rospy.init_node('pixel2depth',anonymous=True)
	    # sub1=rospy.wait_for_message('/mybot/depth_camera/depth/image_raw',Image)
		rospy.Subscriber("/mybot/depth_camera/depth/image_raw", Image,callback=convert_depth_image, queue_size=1)
	    # b=convert_depth_image(sub1)
	    # print(b)
		rospy.spin()


	# def pixel2depth():

	    
	# 	rospy.init_node('pixel2depth',anonymous=True)
	# 	rospy.Subscriber("/mybot/depth_camera/depth/image_raw",Image,callback=convert_depth_image, queue_size=1)
	#     rospy.spin()
	    



		

    
	def pixel_2_world(s,u,v,A,R,t) :
	    k=np.array([[u],[v],[1]])
	    k=s*k
	    # print(k.shape)
	    
	    A_inv=np.linalg.inv(A)
	    # print((A_inv))
	    # print(A_inv.shape)


	    l=np.dot(A_inv,k)
	    # print(l.shape)
	    # print(t)
	    # print(l)

	    m=np.subtract(l,t)
	#     print(m.shape)
	    
	    R_inv=np.linalg.inv(R)
	    # print(R_inv.shape)
	    
	    XYZ=np.dot(R_inv,m)

	    
	    # print(XYZ.shape)
	    print(XYZ)
	    return XYZ


    pixel2depth()
    



u = 262
v = 152
main_func(u,v)



