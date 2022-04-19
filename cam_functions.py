#!usr/bin/env python
# -*- coding:utf-8 _*-
# @author: ycy
# @contact: asuradayuci@gmail.com
# @time: 2019/8/13 下午8:06
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import os
from visualize import visualize
from torchvision.utils import save_image
#from utils.osutils import mkdir_if_missing
from matplotlib import pyplot as plt
import matplotlib.image as mpig
from PIL import Image

dirs_now = os.path.dirname(os.path.abspath(__file__))
PATH = dirs_now + '/mask_eval/'
PATH_EVAL = dirs_now + '/SADouteval/'



def visual_batch(cam, image, b1, b2, b3):
    cam = cam.squeeze()  # b, t, 16, 8                            torch.Size([240, 16, 8])
    cam = cam.reshape(1,1,16,8)
    b,t,h, w = cam.size()  # b, t, 16, 8                            torch.Size([240, 16, 8])

   # image = torch.stack(image, 0).contiguous()  # [bt, 1, 3, 256, 128] torch.Size([240, 1, 3, 256, 128])
    # cam = cam.view(8, 8, *cam.size()[1:])  # 8,8,16,8

    #image = image.view(1, 8, 1, *image.size()[-3:])  # b, t, 1, 3, 256, 128
    image = image.reshape(1,1,1,3,256,128)
    path ="E:/df"
    #mkdir_if_missing(path)
    for i in range(b):  # b
        fig = plt.figure(figsize=(15, 15))
        for j in range(t):  # t
            ax1 = plt.subplot(3, 8, j+1)
            ax1.axis('off')
            plt.title('cam', fontsize=18)
            plt.imshow(cam[i][j].detach().cpu().numpy(), alpha=0.6, cmap='jet')

            ax3 = plt.subplot(3, 8, j + 17)
            ax3.axis('off')
            plt.title('cam+img', fontsize=18)
            cam_ij = cam[i][j].unsqueeze(0)
            cam_ij = cam_ij.unsqueeze(0)
            images_ij = image[i][j]
            heatmap, raw_image = visualize(images_ij, cam_ij)
            heatmap = heatmap.squeeze().cpu().numpy().transpose(1, 2, 0)  #256,128,3
            plt.imshow(heatmap)

            ax4 = plt.subplot(3, 8, j + 9)
            ax4.axis('off')
            plt.title('raw_image', fontsize=18)
            raw_image = raw_image.squeeze().cpu().numpy().transpose(1, 2, 0)
            plt.imshow(raw_image)
        # fig.tight_layout()
        fig.savefig(path + "/iter_{}index_{}_{}.jpg".format(b1, b2, b3))


def get_cam(feature_map, label,weight):
    avg = nn.AdaptiveAvgPool2d((1, 1)) #1024 x 16 x 8
    #y = avg(weight)  # 1024 x 1 x 1
    y = weight.reshape(1024,1,1)
    weight1 = weight.reshape(1024,1,1) # 1 x 1024
    cam =  weight1 * feature_map
    cam = torch.sum(cam,0)
    cam = cam -torch.min(cam)
    cam = cam / torch.max(cam)
    return  cam

def visual_batch_eval(cam, image, length, k):
    cam = torch.stack(cam, 0).contiguous()  # torch.Size([240, 16, 8])
    image = torch.stack(image, 0).contiguous()  # torch.Size([240, 1, 3, 256, 128])

    cam = cam.view(30, 8, 16, -1)  # 8,8,16,8

    image = image.view(30, 8, 1, 3, 256, -1)
    path = PATH_EVAL + "fenzhi{}".format(k)
    #mkdir_if_missing(path)
    for i in range(cam.size(0)):
        fig = plt.figure(figsize=(15, 15))
        for j in range(cam.size(1)):
            ax1 = plt.subplot(3, 8, j+1)
            ax1.axis('off')
            plt.title('cam', fontsize=18)
            plt.imshow(cam[i][j].detach().cpu().numpy(), alpha=0.6, cmap='jet')

            ax3 = plt.subplot(3, 8, j + 17)
            ax3.axis('off')
            plt.title('cam+img', fontsize=18)
            cam_ij = cam[i][j].unsqueeze(0)
            cam_ij = cam_ij.unsqueeze(0)
            images_ij = image[i][j]
            heatmap, raw_image = visualize(images_ij, cam_ij)
            heatmap = heatmap.squeeze().cpu().numpy().transpose(1, 2, 0)
            plt.imshow(heatmap)

            ax4 = plt.subplot(3, 8, j + 9)
            ax4.axis('off')
            plt.title('raw_image', fontsize=18)
            raw_image = raw_image.squeeze().cpu().numpy().transpose(1, 2, 0)
            plt.imshow(raw_image)
        # fig.tight_layout()
        fig.savefig(PATH_EVAL + "fenzhi{}/cambatch_{}.jpg".format(k, i))

    print("hh")



# if __name__ == '__main__':
   #  # PIL-> tensor -> PIL
   #  from torchvision.transforms import ToPILImage, ToTensor
   #
   #  #PIL_img = Image.open('test_pic.jpeg')
   #  #tensor_from_PIL = ToTensor()(PIL_img)
   #  #img = ToPILImage()(tensor_from_PIL)
   #
   #  #img_cv = cv2.imread('3.jpg')
   #
   #  #img_mpig = mpig.imread('1.jpg')
   #  fig = plt.figure(figsize=(15, 15))
   #  img_plt = plt.imread('./pic/0018C1T0003F013.jpg')
   #  # img = torch.from_numpy(img_plt)
   #  img = ToTensor()(img_plt)
   #  img = img.reshape(3,256, 128)
   #  #img1 =img.squeeze().cpu().numpy().reshape(256, 128, 3)
   #  img2 =img.squeeze().cpu().numpy().transpose(1, 2, 0)
   #  #plt.imshow(img1)
   #  #plt.imshow(img2)
   #  #plt.show()
   #  #img = torch.from_numpy(img_plt)
   #  #
   #  #print(img.shape)
   #
   #
   #
   # # print(cam1.size())
   #  #print(cam)
   #  visual_batch(cam1, img, 19, 7, "ours")