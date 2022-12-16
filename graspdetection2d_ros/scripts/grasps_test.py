#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 18:28:12 2021

@author: marco
# Works in the system python environment python 3.8 torch==1.10.1, torchvision==0.11.2, cudatoolkit==10.2, cudnn==7.6, scikit-learn==1.0.2, opencv-python==4.5.4.60
"""
import cv2
import numpy as np
import os
# from inference_by_python import inference #v1
from inference.grasps_pred import get_train_model, GDinfer

import matplotlib.pyplot as plt
from datetime import datetime
from inference.config import color_img_dir, args_network,args_force_cpu,args_use_depth,args_use_rgb,args_n_grasps,args_save


def GDtest(image_size=224): # image_size % 4 should == 0 in this case
    color_img_list = os.listdir(color_img_dir)
    color_img_list.sort()

    net, device =  get_train_model(args_network, args_force_cpu)
    
    for i in range(len(color_img_list)):
        color_img = cv2.imread(color_img_dir+color_img_list[i])[:, :, ::-1] # BGR -> RGB
        
        depth_img = np.ones((image_size,image_size)) # To use real depth images acquired from the camera when testing
        #grasps=inference(args_network, color_img,depth_img,args_use_depth,args_use_rgb, args_n_grasps, args_save,args_force_cpu) # v1
        grasps=GDinfer(net, device, color_img,depth_img,args_use_depth,args_use_rgb, args_n_grasps, args_save)
        # return the grasp in original images 
    
        print("quality,x,y,angle,width of the best grasp in original image:", grasps[0].quality, grasps[0].center[1], grasps[0].center[0], grasps[0].angle, grasps[0].width) # grasps[0].length q,x,y,angle in rad (anti-clockwise is positive),width in 0-100 but useless
        #output example: quality,x,y,angle,width: 0.8242497 75 23 1.4421177 37.71363067626953
        
        #TODO cv2.rectangle
        """
        xo = np.cos(grasps[0].angle)
        yo = np.sin(grasps[0].angle)

        y1 = grasps[0].center[0] + grasps[0].length / 2 * yo
        x1 = grasps[0].center[1] - grasps[0].length / 2 * xo
        y2 = grasps[0].center[0] - grasps[0].length / 2 * yo
        x2 = grasps[0].center[1] + grasps[0].length / 2 * xo
        
        '''
        np.array(
            [
                [y1 - grasps[0].width / 2 * xo, x1 - grasps[0].width / 2 * yo],
                [y2 - grasps[0].width / 2 * xo, x2 - grasps[0].width / 2 * yo],
                [y2 + grasps[0].width / 2 * xo, x2 + grasps[0].width / 2 * yo],
                [y1 + grasps[0].width / 2 * xo, x1 + grasps[0].width / 2 * yo],
            ]
        '''
        """
        time = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
        
        # plot grasps in original images
        fig = plt.figure(figsize=(10, 10))
        plt.ion()
        plt.clf()
        ax = plt.subplot(111)
        ax.imshow(color_img)
        for g in grasps: # for multiple grasps
            g.plot(ax)
        ax.set_title('Grasp')
        ax.axis('on')
        fig.savefig('results/%s_%d_grasp.png' % (time,i))
        
        

if __name__ == "__main__":
    GDtest()
