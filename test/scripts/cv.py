#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
import rospy
import cv_bridge 
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
import math


#scale factor -we set it to 1(default)
test_s=1
#u,v coordinates in pixel frame

test_u=295
test_v=361



def rotation_matrix(a,b,c):

	M=3
	N=3
	# rot_matrix = [ [ 0 for i in range(M) ] for j in range(N) ]
	rot_matrix= np.zeros( (3, 3) )
	# N are no of rows M are no of columns
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



#          x_cam(x)  y_cam(y)  z_cam(z)
# x_world		0		-1(-x)	0
# y_world		0		0		-1(-y)
# z_world		1(z)	0		0

   
#camera matrix R,t - extrinsic

# quad wrt world r,t, matrix
# test_R=[[1 ,0,0],
#         [0 ,1 , 0],
#         [0 , 0 , 1] ]

#camera coordinate sysetem wrt to world is in terms of x,y,z axis is -pi/2,-pi/2,-pi/2 this is wrong
# test_R=rotation_matrix(-np.pi/2,np.pi/8,-np.pi/2)
test_R=[[0 ,-1,0], 
        [0,0 , -1],
        [1, 0 , 0] ]



# test_R=[[0 ,0,1], 
#         [-1,0 , 0],
#         [0 , -1, 0] ]


#test_r2 is fixed and is camera wrt quad rotation matrix

# test_R_2=[[0.9238795325 ,0,0.3826834324],
#         [0 ,1 , 0],
#         [-0.3826834324 , 0 ,0.9238795325] ]

# test_R=np.dot(test_R_1,test_R_2)

# print(test_R)

#wrt camera coordinate system
test_t=[[5], [0],[0]]

# test_t=[[0.0], [0.1226],[-0.05]]
# test_t=[[0.0], [0.1226],[-0.05]]
# test_t=[[0.017500],[-0.056824],[-0.042332]]
# test_t=[[0.017500],[0],[0]]

#wrt camera coordinates of worlds origin (this is wrong -world coordinate system so centre of lens is centr of camra ka model in gazebo so origin of cameera coords syystem is same as cmera model ka centr in gazebo
# test_t=[[0.0], [0.025],[0.0]]

#drone wrt ..world..maybe
# test_t=[[-0.037668], [0.018656],[0.11624]]


#camera matrix - intrinsic
#dont forget to change camera matrix
# camera of drone ka a matrix 
test_A=[[439.0122919979045 ,0.0, 304.5],
        [0.0,  439.0122919979045,   304.5],
        [0.0  ,0.0,1.0]]


# camera of sepaerate test package ka r matrix
# test_A=[[476.7030836014194 ,0.0,400.5],
#         [0.0,  476.7030836014194,   400.5],
#         [0.0  ,0.0,1.0]]

#main function gives us final vector of object in camera coordinate system  z,-x,-y
def main_func(u,v):

	# function convert_depth_image takes input as image and gives out the final vector 
	def convert_depth_image(ros_image):

	    bridge = CvBridge()
	     # Use cv_bridge() to convert the ROS image to OpenCV format

	    try :
	     #Convert the depth image using the default passthrough encoding
	     	depth_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding='passthrough')

	    except :
	     	p=1
		#just random stuff to fill up the except didnt find another way out :)P

	    #Convert the depth image to a Numpy array

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

		
	    C=C*depth_array[v,u]

	    print(C)

	     

	    rospy.loginfo(depth_array)


	#pixel2depth function  will subscribe to /mybot/depth_camera/depth/image_raw" topic and send it to callback function convert_depth image
	def pixel2depth():
	    
		# rospy.init_node('pixel2depth',anonymous=True)
	    # sub1=rospy.wait_for_message('/mybot/depth_camera/depth/image_raw',Image)
		rospy.Subscriber("/front_cam/depth/image_raw", Image,callback=convert_depth_image, queue_size=1)
	    # rospy.Subscriber("/mybot/depth_camera/depth/image_raw", Image,callback=convert_depth_image, queue_size=1)
		# b=convert_depth_image(sub1)
	    # print(b)
		rospy.spin()


	# def pixel2depth():

	    
	# 	rospy.init_node('pixel2depth',anonymous=True)
	# 	rospy.Subscriber("/mybot/depth_camera/depth/image_raw",Image,callback=convert_depth_image, queue_size=1)
	#     rospy.spin()
	    



		

	#pixel_2_world takes all the necessary arguements to convert from pixel coordinates to world coordinates
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
	    
	    # XYZ_DOUBLE_PRIME=np.dot(R_inv,m)
	    # XYZ_PRIME=np.dot(R_inv,m)
	    XYZ=np.dot(R_inv,m)
	    # T=[[0],[0],[0.182466]]
	    # T_PRIME=[[0.05],[0],[-0.06]]
	    # T_DOUBLE_PRIME=[[0],[0],[0]]

	    # R_1=rotation_matrix(0,0,0)
	    # R_PRIME=rotation_matrix(0,np.pi/8,0)
	    # R_DOUBLE_PRIME=rotation_matrix(-np.pi/2,0,-np.pi/2)

	    # R_DOUBLE_PRIME_inv=np.linalg.inv(R_DOUBLE_PRIME)
	    # R_PRIME_inv=np.linalg.inv(R_PRIME)
	    # R_1_inv=np.linalg.inv(R_1)

	    # XYZ_PRIME=np.add((np.dot(R_DOUBLE_PRIME_inv,XYZ_DOUBLE_PRIME)),T_DOUBLE_PRIME)    
	    # XYZ=np.add((np.dot(R_PRIME_inv,XYZ_PRIME)),T_PRIME)
	    # XYZ_WORLD=np.add((np.dot(R_1_inv,XYZ)),T)
	    # XYZ_WORLD=np.add(XYZ,T)

	    
	    # print(XYZ.shape)
	    # print(XYZ)
	    return XYZ
	pixel2depth()



# alpha beta  gamma
# z      y    xs
# yaw  pitch  roll


def rotation_matrix(a,b,c):

	M=3
	N=3
	# rot_matrix = [ [ 0 for i in range(M) ] for j in range(N) ]
	rot_matrix= np.zeros( (3, 3) )
	# N are no of rows M are no of columns
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



def detect(input_image):

	     # Use cv_bridge() to convert the ROS image to OpenCV format
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(input_image, desired_encoding='passthrough')

	


    # Load image, convert to grayscale, and Otsu's threshold 
    # image = cv2.imread('/home/panyam/Downloads/image_raw_screenshot_01.01.2023.png')
    # image = input_image


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 0, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # # Find contours and extract the bounding rectangle coordintes
    # # then find moments to obtain the centroid
    # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    blur = cv2.medianBlur(gray, 25)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,27,6)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilate = cv2.dilate(close, kernel, iterations=2)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)





    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        # Obtain bounding box coordinates and draw rectangle
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)

        # Find center coordinate and draw center point
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(image, (cx, cy), 2, (36,255,12), -1)
        print('Center: ({}, {})'.format(cx,cy))

    cv2.imshow('image', image)
    cv2.waitKey(10000)
    main_func(cx,cy)
    return (cx,cy)



# # rospy.sleep(2)
# print("1")
# rospy.init_node("drone")
# sub=rospy.Subscriber('/front_cam/color/image_raw',Image,detect)
# cv2.imshow('image',sub)
# print("2")





def listener():
    rospy.init_node('drone',anonymous=True)

    rospy.Subscriber("/front_cam/color/image_raw", Image,callback=detect, queue_size=1)

    rospy.spin()




    # rospy.init_node('listener', anonymous=True)

    # rospy.Subscriber("chatter", String, callback)

    # rospy.spin()

if __name__ == '__main__':
    listener()



# detect(sub)

# uv_coordinates=detect(sub1)

# u=uv_coordinates[0]
# v=uv_coordinates[1]


# print(u)
# print(v)






