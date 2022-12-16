#import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
#from PIL import Image

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData     
from utils.visualisation.plot import plot_results, save_results
from utils.dataset_processing.grasp import detect_grasps

from datetime import datetime

import inference.transforms as trans 
from inference.config import image_size, re_size

logging.basicConfig(level=logging.INFO)

'''
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str,
                        help='Path to saved network to evaluate')
    parser.add_argument('--rgb_path', type=str, default='cornell/08/pcd0845r.png',
                        help='RGB Image path')
    parser.add_argument('--depth_path', type=str, default='cornell/08/pcd0845d.tiff',
                        help='Depth Image path')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--save', type=int, default=0,
                        help='Save the results')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args
'''

def get_train_model(args_network,args_force_cpu=False):
    # Load Network
    logging.info('Loading model...')
    net = torch.load(args_network)
    logging.info('Done')
    
    # Get the compute device
    device = get_device(args_force_cpu)
    return net, device


#if __name__ == '__main__':
def GDinfer(net, device, color_img,depth_img,args_use_depth=True,args_use_rgb=True, args_n_grasps=1,args_save=True):
    #args = parse_args()
    
   
    # Record orginal image info
    fh,fw,c= color_img.shape
    s0=max(fh,fw)
    if fh<=fw:
        pad_top0 = int((s0 - fh)/2)
        pad_left0=0
    else: # fh > fw
        pad_top0 = 0
        pad_left0 = int((s0 - fw)/2)
    
    # pre-processing rgb image and depth image0
    pad = (image_size -re_size)//2  # image_size, re_size are imported from config
    color_img_transforms = trans.Compose([
            trans.PaddedSquare('constant'),  # 
            #trans.Resize((image_size, image_size))
            trans.Resize((re_size, re_size)), #    (image_size//2, image_size//2)
            trans.Pad(pad, padding_mode='edge') # image_size//4, padding_mode='edge'
            # trans.Pad(image_size//4, fill=(224,224,224),padding_mode='constant') # obvious border influences results a lot
        ])
    
    color_img = color_img_transforms(color_img)
    #TODO trans.Compose(depth_img)
    
    rgb=color_img
    depth=np.expand_dims(depth_img, axis=2) #
    '''
    logging.info('Loading image...')
    pic = Image.open(args.rgb_path, 'r')
    rgb = np.array(pic)
    pic = Image.open(args.depth_path, 'r')
    depth = np.expand_dims(np.array(pic), axis=2)
    '''
    
    img_data = CameraData(width=image_size,
                          height=image_size,
                          output_size=image_size,
                          include_depth=args_use_depth, 
                          include_rgb=args_use_rgb)

    x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)
#    print(x.size())

    net.eval()
    with torch.no_grad():
        xc = x.to(device)
#        print(xc.size())
        pred = net.predict(xc)

        # post process, gaussian blur and synthesise angle image
        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        #print(q_img.shape)
        
        
        
        if args_save:
            # output grasps and save sub figures in 6 .png files
            grasps=save_results(
                rgb_img=img_data.get_rgb(rgb, False),
                depth_img=np.squeeze(img_data.get_depth(depth)),
                grasp_q_img=q_img,
                grasp_angle_img=ang_img,
                no_grasps=args_n_grasps,
                grasp_width_img=width_img
            )
        else:
            """ # output grasps and save 6 sub figures in a pdf file
            fig = plt.figure(figsize=(10, 10))
            grasps=plot_results(fig=fig,
                                 rgb_img=img_data.get_rgb(rgb, False),
                                 grasp_q_img=q_img,
                                 grasp_angle_img=ang_img,
                                 no_grasps=args_n_grasps,
                                 grasp_width_img=width_img)
            #fig.savefig('img_result.pdf')
            time = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
            fig.savefig('results/{}.img_result.pdf'.format(time))
            """
            grasps = detect_grasps(q_img,ang_img, width_img=width_img, no_grasps=args_n_grasps)
    
    # calculate the grasps in original images 
    #grasps = []
    for i in range(len(grasps)): # for multiple grasps
        x0= round((grasps[i].center[1]-pad)/re_size*s0 - pad_left0)
        y0= round((grasps[i].center[0]-pad)/re_size*s0 - pad_top0)
        grasps[i].center = (int(y0),int(x0))
        #grasps[i].center[1] = (grasps[i].center[1]-pad)/re_size*s0 - pad_left0  
        #grasps[i].center[0] = (grasps[i].center[0]-pad)/re_size*s0 - pad_top0
    
    return grasps
