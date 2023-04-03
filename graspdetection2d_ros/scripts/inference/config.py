#!/usr/bin/env python3
# -*- coding: utf-8 -*-
color_img_dir = 'test_color_img/'

args_network='/home/marco/robotic_sorting/src/graspdetection2d_ros/graspdetection2d_ros/scripts/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
args_force_cpu=False

args_use_depth=True 
args_use_rgb=True
args_n_grasps=1 # 10 for detecting multiple grasps in one image, but GD of Multiple objects is poorer than that of single object
args_save= False #True

# pre-processing rgb image and depth image
image_size=224
re_size = 160
