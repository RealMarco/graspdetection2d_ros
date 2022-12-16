#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 18:24:15 2022

@author: marco
"""
import sys
sys.path.append('/home/marco/catkin_workspace/src/graspdetection2d_ros/graspdetection2d_ros/scripts')
if '/usr/lib/python3/dist-packages' in sys.path: # before importing other modules or packages
    sys.path.remove('/usr/lib/python3/dist-packages')
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

from keypointnet_ros_msgs.msg import Keypoint, Keypoints, KeyObjects

from models.resnet34_classification_paddle import Model_resnet34
from models.keypointnet_deepest_paddle import KeypointNet_Deepest 
from inference.keypoints_pred import KPinfer, PCinfer, get_trained_model
from inference.config import best_PCmodel_path, best_PCmodel_path2, best_KPmodel_path


