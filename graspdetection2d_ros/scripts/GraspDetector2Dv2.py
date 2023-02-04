#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 18:24:15 2022

@author: marco
"""
import sys
sys.path.append('/home/marco/catkin_workspace/src/graspdetection2d_ros/graspdetection2d_ros/scripts')
#if '/usr/lib/python3/dist-packages' in sys.path: # before importing other modules or packages
#    sys.path.remove('/usr/lib/python3/dist-packages')
print (sys.path)

# sys.path.remove('/usr/lib/python3/dist-packages')
# sys.path.remove('/opt/ros/noetic/lib/python3/dist-packages')

import rospy 
import numpy as np 
from threading import Thread

import cv2
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox # , ObjectCount
from std_msgs.msg import String
from sensor_msgs.msg import Image # ?
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
import tf

from graspdetection2d_ros_msgs.msg import Grasp, Grasps, Grasp2DObjects, GraspPoseClass

from inference.grasps_pred import get_train_model, GDinfer
from inference.config import args_network,args_force_cpu,args_use_depth,args_use_rgb,args_n_grasps,args_save, image_size

# References:
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
# https://gitlab.com/neutron-nuaa/robot-arm/-/tree/main/Paper_manipulation_and_shoe_packaging/darknet_ros
# https://www.paddlepaddle.org.cn/documentation/docs
# http://wiki.ros.org/rospy
# http://wiki.ros.org/rospy_tutorials/Tutorials/
# https://docs.ros.org/en/melodic/api/rospy/html/
# https://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html

# define a color image callback function
def cimgCallback(cimg):
#    bridge =  CvBridge()
    try:
        cv_cimg = cvbridge.imgmsg_to_cv2(cimg, "rgb8") # "bgr8", desired_encoding="passthrough"
    except CvBridgeError as e:
        rospy.logerr('Converting color image error: ' + str(e))
    global ColorImage # essential to claim a variable as global 
    ColorImage = cv_cimg
    rospy.loginfo("Subscribing a color image from camera with a shape of (%d, %d, %d) "%(ColorImage.shape[0],ColorImage.shape[1],ColorImage.shape[2])) # (H,W,C)
    
# define a raw color image subscriber
def color_img_subscriber():
#    rospy.init_node('color_img_subscriber', anonymous=True)
    rospy.Subscriber('/camera/color/image_raw', Image, cimgCallback, queue_size=1) # '/camera/image_raw'
#    rospy.spin()
    
# define a depth image callback function
def dimgCallback(dimg):
    try:
        cv_dimg = cvbridge.imgmsg_to_cv2(dimg, "mono16") #TODO check if it is unit16, desired_encoding="passthrough"
    except CvBridgeError as e:
        rospy.logerr('Converting depth image error: ' + str(e))
    global DepthImage
    DepthImage = cv_dimg
    rospy.loginfo("Subscribing a depth image from camera with a shape of ", DepthImage.shape) # (H,W,C)

# define a depth image subscriber
def depth_img_subscriber():
    rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, dimgCallback, queue_size=1)

# callback function of Camera info
def camInfoCallback(camera_ip):
    global camera_info
    camera_info = camera_ip
    
# subscriber of Camera info 
def camera_info_subscriber():
    rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, camInfoCallback, queue_size=1)

# define an infer callback function, which would cropping ColorImage by bbxes firtly
def inferCallback(bbxes):
    #rospy.sleep(0.2)
    global ObjectBbxes, CroppedImgs, CroppedXYmin, CroppedXYmax, CroppedCategories # not required to claim a list global
    ObjectBbxes.clear() # clear the existing ObjectBbxes; global ObjectBbxes ObjectBbxes=[]
    best_prob = 0
    [xmin, ymin, xmax, ymax, category] = [0,0,0,0,'']
    for bbx in bbxes.bounding_boxes:
        thresh_prob = 0.5
        #if bbx.probability > thresh_prob : # output multiple grasps of the objects whose probability is higher than threshold
        if bbx.probability > best_prob :  # output the grasp of object with highest object probability
            best_prob = bbx.probability
            #[xmin, ymin, xmax, ymax] = [round(bbx.xmin), round(bbx.ymin), round(bbx.xmax), round(bbx.ymax)]
            # crop slightly bigger to improve the predictions of Grasps
            [xmin, ymin, xmax, ymax, category] = [round(bbx.xmin*0.98), round(bbx.ymin*0.98), round(bbx.xmax*1.02), round(bbx.ymax*1.02), bbx.Class]
            
            '''# output multiple grasps of the objects whose probability is higher than threshold
            ObjectBbxes.append([xmin, ymin, xmax, ymax,category]) # ObjectBbxes only has one item with highest probability in this case 
            rospy.loginfo("Subscribing an object bounding box: xmin:%d ymin:%d xmax:%d ymax:%d category: %s"
                  %(xmin, ymin, xmax, ymax, category))
            '''
            
    # output the grasp of object with highest object probability
    if [xmin, ymin, xmax, ymax, category] != [0,0,0,0,'']:
        ObjectBbxes.append([xmin, ymin, xmax, ymax, category]) # ObjectBbxes only has one item with highest probability in this case 
        rospy.loginfo("Subscribing an object bounding box: xmin:%d ymin:%d xmax:%d ymax:%d category: %s"
              %(xmin, ymin, xmax, ymax, category))
    
    
    if ObjectBbxes != []: # avoid grasp inference and publishment when no object was detected
        # crop original image by bounding boxes
        CroppedImgs.clear() # clear the existing CroppedImgs
        CroppedXYmin.clear()
        CroppedXYmax.clear()
        CroppedCategories.clear()
        for ObjectBbx in ObjectBbxes:
    #        global ColorImage, CroppedImgs
            cropped_img = ColorImage[ObjectBbx[1]:ObjectBbx[3], ObjectBbx[0]:ObjectBbx[2]]# [ymin:ymax,xmin:xmax]
            CroppedImgs.append(cropped_img)
            CroppedXYmin.append([ObjectBbx[0], ObjectBbx[1]])
            CroppedXYmax.append([ObjectBbx[2], ObjectBbx[3]])
            CroppedCategories.append(ObjectBbx[4])
            
            
        # Infer CroppedImgs list and return grasps.
        global gobjects_list, net, device 
        ### May use multi threads to speed up the inference
        ### Infer Grasps
        gobjects_list.clear()
        
        depth_img = np.ones((image_size,image_size)) # To use real depth images acquired from the camera when testing
        
        for CroppedImg in CroppedImgs:
            gps=GDinfer(net, device, CroppedImg ,depth_img,args_use_depth,args_use_rgb, args_n_grasps, args_save)
            gobjects_list.append(gps) # grasps
        
        # publish grasps with its GDImage
        #rospy.sleep(0.2)
        grasps_publisher()

# define a BoundingBoxes subscriber
def bbxes_subscriber():
    # initialize ros node 
#    rospy.init_node('bbxes_subscriber', anonymous=True)
    
    # Registration: create a subscriber, and subscribing a topic named bounding_boxes with BoundingBoxes message
    # Register bbxes Callback function
    rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, inferCallback, queue_size=1) # 'bounding_boxes'
    
    # recurrently subscribe the bbxes
#    rospy.spin() # blocking function

#define an object detection image (marked with bouding boxes) callback function
def odImgCallback(odImg):
#    bridge =  CvBridge()
    try:
        cv_odImg = cvbridge.imgmsg_to_cv2(odImg, "rgb8") # bgr8, desired_encoding="passthrough"
    except CvBridgeError as e:
        rospy.logerr('Converting object detection image error: ' + str(e))
    global ODImage # essential to claim a variable as global 
    ODImage = cv_odImg
    rospy.loginfo("Subscribing an object detection image with a shape of (%d, %d, %d)"%(ODImage.shape[0],ODImage.shape[1], ODImage.shape[2])) # (H,W,C)

# define an object detection image subscriber
def od_img_subscriber():
#    rospy.init_node('bbx_img_subscriber', anonymous=True)
    rospy.Subscriber('/darknet_ros/detection_image', Image, odImgCallback, queue_size=1) # '/camera/image_raw'
#    rospy.spin()

# transform from pixel coordinate to camera coordinate
def coordinateMap(grasp_x,grasp_y):
    global DepthImage, camera_info
    real_z = 0.001*DepthImage[grasp_y][grasp_x]; #TODO check if the parameter 0.001 is accurate
    # calculate x, y via Camera internal parameter matrix
    real_x = (grasp_x - camera_info.K[2])/camera_info.K[0]*real_z
    real_y = (grasp_y - camera_info.K[5])/camera_info.K[4]*real_z
    
    outputPoint=PoseStamped()
    outputPoint.header.frame_id = "/camera_depth_optical_frame"
    outputPoint.header.stamp = rospy.Time.now # now(), timestamp
    outputPoint.point.x = real_x
    outputPoint.point.y = real_y
    outputPoint.point.z = real_z
    
    return outputPoint

# define a grasps, object state publisher, orientation publisher, publish 
def grasps_publisher():
    # init ros node
#    rospy.init_node('object_state_publiser', anonymous=True) # try anonymous person
    try:
        # rospy.sleep(0.1) # wait for finishing node registration, or the 1st msg wouldn't be published
        # if len(state_classes)==len(confident_kps):
        # create msg
        global ObjectBbxes, GDImage, ColorImage, CroppedXYmin, CroppedXYmax, CroppedCategories, gobjects_list #ODImage
        GDImage = ColorImage #ODImage
        GObjects = Grasp2DObjects()  # 
        for i in range(len(gobjects_list)): # gobjects_list[i] is grasps
            grasps =  Grasps()
            for j in range(len(gobjects_list[i])): #  gobjects_list[i][j] is grasp
                grasp = Grasp()
                grasp.quality =  gobjects_list[i][j].quality 
                
                grasp.x = round(gobjects_list[i][j].center[1] + CroppedXYmin[i][0])
                grasp.y = round(gobjects_list[i][j].center[0] + CroppedXYmin[i][1])
                grasp.angle = gobjects_list[i][j].angle # in rad
                grasp.width = gobjects_list[i][j].width
                #print(grasp.angle)
                grasps.grasps.append(grasp)
                grasps.Class = CroppedCategories[i]
                
                #visualize grasp rectangle
                rot_rectangle = ((grasp.x, grasp.y), (100, 50), -1*grasp.angle/np.pi*180)
                box = cv2.boxPoints(rot_rectangle) 
                box = np.int0(box) #Convert into integer values
                GDImage = cv2.drawContours(GDImage,[box],0,g_colors[j],2)
                GDImage =cv2.circle(GDImage, (grasp.x, grasp.y), 8, g_colors[j], -1) # kp_colors[j], circle(img, point_center, radius, RGB, thickness
                
            GObjects.gobjects.append(grasps)
        
        # Publishing the best graspPose of the object with highest confidence score in PoseStamped format
        grasp = GObjects.gobjects[0].grasps[0] #GObjects[0][0]
        
        ''' # publish GraspPoseClass messages
        grasp_pose_class = GraspPoseClass()
        grasp_pose_class.grasp_pose.header.frame_id = "/camera_color_optical_frame" #"camera_color_frame"
        grasp_pose_class.grasp_pose.header.stamp = rospy.Time()
        grasp_pose_class.grasp_pose.pose.position = coordinateMap(grasp.x, grasp.y).point
        grasp_pose_class.grasp_pose.pose.orientation = tf.createQuaternionMsgFromYaw(grasp.angle) # TODO from Grasp.angle to the yaw of end_effector
        grasp_pose_class.Class = GObjects.gobjects[0].Class
        grasp_pose_class_publisher.publish(grasp_pose_class)
        '''
        
        # publish PoseStamped and String (Class) messages
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = "/camera_color_optical_frame" #"camera_color_frame"
        grasp_pose.header.stamp = rospy.Time()
        grasp_pose.pose.position = coordinateMap(grasp.x, grasp.y).point
        grasp_pose.pose.orientation = tf.createQuaternionMsgFromYaw(grasp.angle) # TODO from Grasp.angle to the yaw of end_effector
        grasp_pose_publisher.publish(grasp_pose)
        rospy.loginfo("Publishing the best grasp pose of the object with highest confidence score")
        grasp_class = String()
        grasp_class.data = GObjects.gobjects[0].Class #from str to std_msgs.String
        grasp_class_publisher.publish(grasp_class)
        rospy.loginfo("Publishing the object class of the best grasp pose")
        
        
        # publish grasps of objects
        obj_grasps_publisher.publish(GObjects)
        rospy.loginfo("Publishing the grasps of %d objects"%(len(GObjects.gobjects)))
        
        # publisher Present the grasps on ODImage, similar to the topic /darknet_ros/detection_image in object detection
        # GDImage =  np.asarray(GDImage)
        cv2.imshow("Grasp Detection", GDImage[:, :, ::-1]) # RGB to BGR
        while (cv2.waitKey(30)==27):
            pass
        GDImage_msg = cvbridge.cv2_to_imgmsg(GDImage, "bgr8") #"bgr8", encoding="passthrough"
        gd_img_publisher.publish(GDImage_msg)
        rospy.loginfo("Publishing an grasp detection image with a shape of (%d, %d, %d)"%(GDImage.shape[0],GDImage.shape[1],GDImage.shape[2])) 
    except rospy.ROSInterruptException: # except [error type]
        pass 


if __name__ == "__main__":# avoid automatic running below lines when this .py file is imported by others.
#    try:
    ColorImage = np.zeros((2,2,3))
    DepthImage = np.zeros((2,2,3))
    camera_info = CameraInfo() 
    # ODImage  = np.zeros((2,2)) 
    GDImage =  np.ones((2,2,3))
    ObjectBbxes = []
    CroppedImgs = []
    CroppedXYmin = []  # record the [[xmin, ymin],] for coordinate transferring 
    CroppedXYmax = []
    CroppedCategories = [] # record the object Class for each bbx
    gobjects_list = []
    g_colors = [(0,0,255),    (0,255,0),  (255,0,0),   
                (255,255,0),  (255,0,255),  (0,255,255),
                (255,128,128),(128,255,128),(128,128,255),(255,255,255)] 

    cvbridge = CvBridge()
    
    net, device =  get_train_model(args_network, args_force_cpu)
    
    rospy.init_node("graspdetection2d_ros", anonymous=True)
    #rospy.init_node('graspdetection2d_ros')
    # Registration: Create a publisher, and publish a topic named person_info with test_topic.msg.Person message, queue size =4
    obj_grasps_publisher = rospy.Publisher('/graspdetection2d_ros/object_grasps', Grasp2DObjects, queue_size=1) # latch =
    gd_img_publisher = rospy.Publisher('/graspdetection2d_ros/gd_image', Image, queue_size=1)
    
    #rospy.init_node("object_graspPose", anonymous=True)
    grasp_pose_publisher = rospy.Publisher('/object_graspPose', PoseStamped, queue_size=1) # '/graspdetection2d_ros/object_graspPose'
    grasp_class_publisher= rospy.Publisher('/object_graspClass', String, queue_size=1)
    #grasp_class_publisher= rospy.Publisher('/object_graspClass', str, queue_size=1)
    # grasp_pose_class_publisher = rospy.Publisher('/object_graspPoseClass', GraspPoseClass, queue_size=1)
    
    rospy.sleep(0.2) # wait for finishing node registration, or the 1st msg wouldn't be published
    
    # keep subscribing... cyclically
    t1 = Thread(target=color_img_subscriber) # args=(arg_for_target_function,)
    t4 = Thread(target=depth_img_subscriber)
    t5 = Thread(target=camera_info_subscriber) # TODO subscribe once or everytime
    t2 = Thread(target=bbxes_subscriber)  # subscribe bbxes and infer keypoinis + state class, then publish grasps, GDImage
    #t3 = Thread(target=od_img_subscriber)
    t1.start()
    t4.start()
    t5.start()
    rospy.sleep(0.2) # wait for color_img_subscriber
    #t3.start()
    t2.start()
    
#        # lock CroppedImgs, ColorImage, ODImage when t2 is running
#        .join()
#        lock = threading.Lock()
    rospy.spin()
    
    # Recurrently do publishing or publish in inferCallback
        
#    except rospy.ROSInterruptException: # except [error type]
#        pass 
