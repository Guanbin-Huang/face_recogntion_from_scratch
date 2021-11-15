import cv2 
import numpy as np
import torch


def scale_coord(dst_shape, xyrb, src_shape):
    """ 
    xyrb on dst
    dst ---> src
     """
    xyrb_1 = torch.vstack((torch.unsqueeze(xyrb,dim = 0).reshape(2,2).T, torch.ones(1,2)))

    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]
    
    scale = min(dst_h / src_h, dst_w / src_w) 

    translation_w = -src_w*0.5*scale + dst_w*0.5
    translation_h = -src_h*0.5*scale + dst_h*0.5

    M = torch.tensor([[scale, 0, translation_w], # M  src --> dst
                      [0, scale, translation_h]])
                      
    inv_M = torch.tensor(cv2.invertAffineTransform(M.numpy()))
    
    return inv_M @ xyrb_1  # 2x2 = 2x3  @  3 x 2



if __name__ == "__main__":  
    coord = torch.tensor([378.06540, 237.03140, 509.64566, 415.10995])
    img1_shape = torch.tensor([800, 672]) # net
    img0_shape = torch.tensor([635, 494]) # raw_img

    scale_coord(img1_shape, coord, img0_shape)



