# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import torchvision.transforms as transforms
import torchvision
import math
from misc import *
from collections import namedtuple
import os
import torchvision.transforms.functional as TF

# This sets the batch norm layers in pytorch as if {'is_training': False, 'scale': True} in tensorflow
def bn_init_as_tf(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = True  # These two lines enable using stats (moving mean and var) loaded from pretrained model
        m.eval()  # or zero mean and variance of one if the batch norm layer has no pretrained values
        m.affine = True
        m.requires_grad = True


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


class edge_loss_test(nn.Module):
    def __init__(self, variance):
        super(edge_loss_test, self).__init__()
        self.beta = variance

    def forward(self,image, depth_est, depth_gt, mask,step,writer):
        # is_vis = False
        # diff
        depth_diff = torch.abs(depth_est - depth_gt)
        # 先对他们做边缘检测，测试用

        # 提取的mask，目前还没有用
        # mask = grad_mask(mask)
        # 设置路径
        # sobel_edge_dir = './vis_edge_sobel/'
        # if not os.path.exists(sobel_edge_dir):
        #     os.mkdir(sobel_edge_dir)
        # 对image进行反归一化，隐射到0-1也好可视化
        image_nor = unnormalize_tensor(image.clone())
        # print(torch.sum(torch.isnan(image_nor)).item(), torch.sum(torch.isnan(image_nor)).item())

        # 变成灰度图像
        gray_tensor = 0.299 * image_nor[:, 0, :, :] + 0.587 * image_nor[:, 1, :, :] + 0.114 * image_nor[:, 2, :, :]
        # 将图像的范围从[0, 1]调整到[0, 255]
        gray_tensor = gray_tensor.unsqueeze(1)
        # print(torch.sum(torch.isnan(gray_tensor)).item(), torch.sum(torch.isnan(gray_tensor)).item())

        # 保存图像
        # RGB
        # if is_vis:
        # image_orin = image_nor.detach().cpu().numpy().squeeze()*255
        # img_plt = np.transpose(image_orin, (1, 2, 0)).astype(np.uint8)
        # plt.imsave(sobel_edge_dir+'output_image_nor.jpg', img_plt)
        # 灰度
        # gray_images = (gray_tensor.detach().cpu().numpy().squeeze() *255).astype(np.uint8)
        # print("grayimage",type(gray_images),np.max(gray_images),gray_images.shape)
        # cv2.imwrite(sobel_edge_dir+"gray.jpg",gray_images)
        # sobel算子
        # sobel_est_edge = sobel_misc(depth_est)
        sobel_gt_edge = sobel_misc(depth_gt)
        sobel_image_edge = sobel_misc(gray_tensor)
        #这两个没有nan 和 inf

        # 提前归一化 sobel
        # sgt_nor = normalize_to_01(sobel_gt_edge)
        # save_one("sgt_sobelnor.jpg",sgt_nor)
        # simage_nor = normalize_to_01(sobel_image_edge)
        # save_one("simage_sobelnor.jpg",simage_nor)

        # sobel_image_edge_di = dilate_edges(sobel_image_edge)
        # save_one(sobel_image_edge_di,sobel_edge_dir+f"{step}.jpg")
        # sobel_edge_path = [sobel_edge_dir+f'{step}est.jpg',sobel_edge_dir+f'{step}gt.jpg',sobel_edge_dir+f'{step}image.jpg',sobel_edge_dir+f'{step}image_nor.jpg']
        # color(sobel_est_edge,0,10,'gray',sobel_edge_path[0])
        # color(sobel_gt_edge,0,10,'gray',sobel_edge_path[1])
        # color(sobel_image_edge,0,10,'gray',sobel_edge_path[2])
        # color(sobel_image_edge_di,0,10,'gray',sobel_edge_path[3])

        #
        # canny_est_edge = canny_edge_detection(depth_est)
        # canny_gt_edge = canny_edge_detection(depth_gt)
        # canny_image_edge = canny_edge_detection(gray_tensor)
        # canny_edge_path = [sobel_edge_dir+f'{step}cest.jpg',sobel_edge_dir+f'{step}cgt.jpg',sobel_edge_dir+f'{step}cimage.jpg',sobel_edge_dir+f'{step}cimage_nor.jpg']
        # color(canny_est_edge,0,10,'gray',canny_edge_path[0])

        # sgt_nor = normalize_to_01(canny_gt_edge)
        # save_one("sgt_cannynor.jpg",sgt_nor)
        # simage_nor = normalize_to_01(canny_image_edge)
        # save_one("simage_cannynor.jpg",simage_nor)
        # color(canny_gt_edge,0,10,'gray',canny_edge_path[1])
        # color(canny_image_edge,0,10,'gray',canny_edge_path[2])
        #
        # LoG_gt_edge = LoG_edge_detection(depth_gt)
        # LoG_image_edge = LoG_edge_detection(gray_tensor)
        # LoG_edge_path = [sobel_edge_dir+f'{step}lest.jpg',sobel_edge_dir+f'{step}lgt.jpg',sobel_edge_dir+f'{step}limage.jpg',sobel_edge_dir+f'{step}cimage_nor.jpg']
        # color(canny_est_edge,0,10,'gray',canny_edge_path[0])
        # color(LoG_gt_edge,0,10,'gray',LoG_edge_path[1])
        # color(LoG_image_edge,0,10,'gray',LoG_edge_path[2])

        # sobel_edeg_value = [sobel_est_edge,sobel_gt_edge,sobel_image_edge]

        # apply_otsu_thresholding(sobel_image_edge, sobel_edge_dir+'apply_otsu_thresholding.png')
        # mask
        # sobel_edge_dir = './vis_edge_sobel/'
        # if not os.path.exists(sobel_edge_dir):
        #     os.mkdir(sobel_edge_dir)
        # sobel_est_edge[mask==False] = 0
        # sobel_gt_edge[mask==False] = 0
        # sobel_image_edge[mask==False] = 0
        # sobel_edge_mask_path = [sobel_edge_dir+'est_mask.jpg',sobel_edge_dir+'gt_mask.jpg',sobel_edge_dir+'image_mask.jpg']
        # # colors([sobel_est_edge,sobel_gt_edge,sobel_image_edge],0,10,'gray',sobel_edge_mask_path)
        # color(sobel_est_edge,0,10,'gray',sobel_edge_mask_path[0])
        # color(sobel_gt_edge,0,10,'gray',sobel_edge_mask_path[1])
        # color(sobel_image_edge,0,10,'gray',sobel_edge_mask_path[2])
        # mask结束
        # 或者sobel掩码
        # edge_mask1 = torch.where(sobel_image_edge > 0.5, torch.tensor(1.0,device=image.device), torch.tensor(0.0,device=image.device))
        # 没有归一化
        edge_masksobel1 = torch.where(sobel_image_edge > 0.5, torch.tensor(1.0,device=image.device), torch.tensor(0.0,device=image.device))
        edge_masksobelgt1 = torch.where(sobel_gt_edge > 0.5, torch.tensor(1.0,device=sobel_gt_edge.device), torch.tensor(0.0,device=sobel_gt_edge.device))
        # print(edge_masksobelgt1.shape,edge_masksobel1.shape)
        # print(torch.sum(torch.isnan(edge_masksobel1)).item(), torch.sum(torch.isnan(edge_masksobel1)).item())
        # print(torch.sum(torch.isnan(edge_masksobelgt1)).item(), torch.sum(torch.isnan(edge_masksobelgt1)).item())

        # edge_maskcanny1 = torch.where(canny_image_edge > 0.5, torch.tensor(1.0,device=image.device), torch.tensor(0.0,device=image.device))
        # edge_maskcannygt1 = torch.where(canny_gt_edge > 0.5, torch.tensor(1.0,device=sobel_gt_edge.device), torch.tensor(0.0,device=sobel_gt_edge.device))
        # vis_edge1(edge_masksobelgt1,edge_masksobel1,edge_maskcanny1,edge_maskcannygt1,'v1.png')
        # 归一化后阈值0.7
        # edge_masksobel_nor1 = torch.where(sgt_nor > 0.5, torch.tensor(1.0,device=image.device), torch.tensor(0.0,device=image.device))
        # edge_masksobelgt_nor1 = torch.where(simage_nor > 0.5, torch.tensor(1.0,device=sobel_gt_edge.device), torch.tensor(0.0,device=sobel_gt_edge.device))
        # edge_maskcanny_nor1 = torch.where( > 0.5, torch.tensor(1.0,device=image.device), torch.tensor(0.0,device=image.device))
        # edge_maskcannygt_nor1 = torch.where(canny_gt_edge > 0.5, torch.tensor(1.0,device=sobel_gt_edge.device), torch.tensor(0.0,device=sobel_gt_edge.device))
        # vis_edge1(edge_masksobelgt_nor1,edge_masksobel_nor1,edge_maskcanny_nor1,edge_maskcannygt_nor1,'v2.png')

        # edge_masksobel2 = torch.where(sobel_image_edge > 0.65, torch.tensor(1.0,device=image.device), torch.tensor(0.0,device=image.device))
        # edge_masksobelgt2 = torch.where(sobel_gt_edge > 0.65, torch.tensor(1.0,device=sobel_gt_edge.device), torch.tensor(0.0,device=sobel_gt_edge.device))
        # edge_maskcanny2 = torch.where(canny_image_edge > 0.65, torch.tensor(1.0,device=image.device), torch.tensor(0.0,device=image.device))
        # edge_maskcannygt2 = torch.where(canny_gt_edge > 0.65, torch.tensor(1.0,device=sobel_gt_edge.device), torch.tensor(0.0,device=sobel_gt_edge.device))
        # vis_edge1(edge_masksobelgt2,edge_masksobel2,edge_maskcanny2,edge_maskcannygt2,'v3.png')

        # # 归一化后阈值0.5
        # edge_masksobel_nor2 = torch.where(sobel_image_edge > 0.65, torch.tensor(1.0,device=image.device), torch.tensor(0.0,device=image.device))
        # edge_masksobelgt_nor2 = torch.where(sobel_gt_edge > 0.65, torch.tensor(1.0,device=sobel_gt_edge.device), torch.tensor(0.0,device=sobel_gt_edge.device))
        # edge_maskcanny_nor2 = torch.where(canny_image_edge > 0.65, torch.tensor(1.0,device=image.device), torch.tensor(0.0,device=image.device))
        # edge_maskcannygt_nor2 = torch.where(canny_gt_edge > 0.65, torch.tensor(1.0,device=sobel_gt_edge.device), torch.tensor(0.0,device=sobel_gt_edge.device))
        # vis_edge1(edge_masksobelgt_nor2,edge_masksobel_nor2,edge_maskcanny_nor2,edge_maskcannygt_nor2,'v4.png')



        # apply_otsu_thresholding(edge_masksobelgt,"edge_masksobel_otsu.jpg")
        # apply_otsu_thresholding(edge_maskcannygt,"edge_maskcanny_otsu.jpg")
        # plt.imsave("sobel_edge_bina.jpg", edge_masksobel.squeeze().cpu().numpy(), cmap='gray')
        # plt.imsave("sobel_gt_edge_bina.jpg",edge_masksobelgt.squeeze().cpu().numpy(), cmap='gray')
        # plt.imsave("canny_edge_bina.jpg",edge_maskcanny.squeeze().cpu().numpy(), cmap='gray')
        # plt.imsave("cannygt_edge_bina.jpg",edge_maskcannygt.squeeze().cpu().numpy(), cmap='gray')

        edge_masksobelgt_di = dilate_edges(edge_masksobelgt1)
        # print(torch.sum(torch.isnan(edge_masksobel1)).item(), torch.sum(torch.isnan(edge_masksobel1)).item())

        # plt.imsave("sobel_gt_edge_bina_di.jpg",edge_masksobelgt.squeeze().cpu().numpy(), cmap='gray')
        # edge_maskcannygt = dilate_edges(edge_maskcannygt_nor1)
        # plt.imsave("canny_gt_edge_bina_di.jpg", edge_maskcannygt.squeeze().cpu().numpy(), cmap='gray')

        # edge_maskc = torch.where( > 0.65, torch.tensor(1.0,device=image.device), torch.tensor(0.0,device=image.device))
        # edge_maskl = torch.where(sobel_image_edge > 0.65, torch.tensor(1.0,device=image.device), torch.tensor(0.0,device=image.device))

        # edge_mask = torch.where(sobel_image_edge > 0.8, torch.tensor(1.0,device=image.device), torch.tensor(0.0,device=image.device))
        # print("edge_mask2",type(edge_mask2),torch.max(edge_mask2),torch.min(edge_mask2),torch.mean(edge_mask2),edge_mask2.shape[2]*edge_mask2.shape[3])
        b,_,h2,w2 = edge_masksobelgt1.shape
        mask_vaild_ = torch.zeros_like(edge_masksobelgt1)
        mask_vaild_[...,10:h2-10,10:w2-10] = 1
        # 统计等于1的元素个数

        edge_masksobel = ((edge_masksobel1 == 1) & (edge_masksobelgt_di == 1)& (mask_vaild_ == 1)).to(torch.float32)
        # print(torch.sum(torch.isnan(edge_masksobel)).item(), torch.sum(torch.isnan(edge_masksobel)).item())

        for i in range(b):
            writer.add_image('edge_masksobel/image/{}'.format(i), (edge_masksobel[i, :, :, :].data),
                             step)
        # edge_maskcanny = ((edge_maskcannygt_nor1 == 1) & (edge_maskcannygt == 1)& (mask_vaild_ == 1)).to(torch.float32)

        # edge_mask_np2 = edge_masksobel.squeeze().detach().cpu().numpy()*255
        # plt.imsave("edge_mask_sobel.png",edge_mask_np2, cmap='gray')
        # edge_mask_np2 = edge_maskcanny.squeeze().detach().cpu().numpy()*255
        # plt.imsave("edge_mask_canny.png",edge_mask_np2, cmap='gray')
        # tile_ = [['sobel gt','sobel image','canny gt','canny image'],['sobel dilate_edges','canny dilate_edges'],['sobel mask','canny mask']]
        # vis_edge2(edge_masksobelgt_nor1,edge_masksobel_nor1,edge_maskcannygt_nor1,edge_maskcanny_nor1,
        #           edge_masksobelgt,edge_maskcannygt,
        #           edge_masksobel,edge_maskcanny,
        #           tile_ ,'vv2.png')

        # 可视化掩码
        # vis_edge_mask(edge_mask1,edge_mask2,edge_mask)
        # vis_edge_mask(None,edge_mask2,None)

        weighted_depth_diff = self.beta * depth_diff * edge_masksobel + (1 - self.beta) * depth_diff * (1 - edge_masksobel)
        focal_loss = torch.mean(weighted_depth_diff) * 0.8


        return focal_loss

class edge_loss(nn.Module):
    def __init__(self, variance):
        super(edge_loss, self).__init__()
        self.beta = variance

    def forward(self,image, depth_est, depth_gt, mask):
        # is_vis = False
        # diff
        depth_diff = torch.abs(depth_est - depth_gt)
        # 先对他们做边缘检测，测试用

        # 提取的mask，目前还没有用
        # mask = grad_mask(mask)
        # 设置路径
        # sobel_edge_dir = './vis_edge_sobel/'
        # if not os.path.exists(sobel_edge_dir):
        #     os.mkdir(sobel_edge_dir)
        # 对image进行反归一化，隐射到0-1也好可视化
        image_nor = unnormalize_tensor(image.clone())
        # 变成灰度图像
        gray_tensor = 0.299 * image_nor[:, 0, :, :] + 0.587 * image_nor[:, 1, :, :] + 0.114 * image_nor[:, 2, :, :]
        # 将图像的范围从[0, 1]调整到[0, 255]
        gray_tensor = gray_tensor.unsqueeze(1)
        # 保存图像
        # RGB
        # if is_vis:
        #     image_orin = image_nor.detach().cpu().numpy().squeeze()*255
        #     img_plt = np.transpose(image_orin, (1, 2, 0)).astype(np.uint8)
        #     plt.imsave(sobel_edge_dir+'output_image_nor.jpg', img_plt)
        #     # 灰度
        #     gray_images = (gray_tensor.detach().cpu().numpy().squeeze() *255).astype(np.uint8)
        #     # print("grayimage",type(gray_images),np.max(gray_images),gray_images.shape)
        #     cv2.imwrite(sobel_edge_dir+"g.jpg",gray_images)
        #     sobel_est_edge = sobel_misc(depth_est)
        #     sobel_gt_edge = sobel_misc(depth_gt)


        sobel_image_edge = sobel_misc(gray_tensor)
        #
        # sobel_edeg_value = [sobel_est_edge,sobel_gt_edge,sobel_image_edge]
        # sobel_edge_path = [sobel_edge_dir+'est.jpg',sobel_edge_dir+'gt.jpg',sobel_edge_dir+'image.jpg',sobel_edge_dir+'image_nor.jpg']
        # color(sobel_est_edge,0,10,'gray',sobel_edge_path[0])
        # color(sobel_gt_edge,0,10,'gray',sobel_edge_path[1])
        # color(sobel_image_edge,0,10,'gray',sobel_edge_path[2])
        # apply_otsu_thresholding(sobel_image_edge, sobel_edge_dir+'apply_otsu_thresholding.png')
        # mask
        # sobel_edge_dir = './vis_edge_sobel/'
        # if not os.path.exists(sobel_edge_dir):
        #     os.mkdir(sobel_edge_dir)
        # sobel_est_edge[mask==False] = 0
        # sobel_gt_edge[mask==False] = 0
        # sobel_image_edge[mask==False] = 0
        # sobel_edge_mask_path = [sobel_edge_dir+'est_mask.jpg',sobel_edge_dir+'gt_mask.jpg',sobel_edge_dir+'image_mask.jpg']
        # # colors([sobel_est_edge,sobel_gt_edge,sobel_image_edge],0,10,'gray',sobel_edge_mask_path)
        # color(sobel_est_edge,0,10,'gray',sobel_edge_mask_path[0])
        # color(sobel_gt_edge,0,10,'gray',sobel_edge_mask_path[1])
        # color(sobel_image_edge,0,10,'gray',sobel_edge_mask_path[2])
        # mask结束
        # 或者sobel掩码
        # edge_mask1 = torch.where(sobel_image_edge > 0.5, torch.tensor(1.0,device=image.device), torch.tensor(0.0,device=image.device))
        edge_mask2 = torch.where(sobel_image_edge > 0.65, torch.tensor(1.0,device=image.device), torch.tensor(0.0,device=image.device))
        # edge_mask = torch.where(sobel_image_edge > 0.8, torch.tensor(1.0,device=image.device), torch.tensor(0.0,device=image.device))
        # print("edge_mask2",type(edge_mask2),torch.max(edge_mask2),torch.min(edge_mask2),torch.mean(edge_mask2),edge_mask2.shape[2]*edge_mask2.shape[3])
        _,_,h2,w2 = edge_mask2.shape
        mask_vaild_ = torch.zeros_like(edge_mask2)
        mask_vaild_[...,10:h2-10,10:w2-10] = 1
        # 统计等于1的元素个数

        edge_mask2 = ((edge_mask2 == 1) & (mask_vaild_ == 1)).to(torch.float32)
        # edge_mask_np2 = edge_mask2.squeeze().detach().cpu().numpy()*255
        # plt.imsave("edge_mask_np2.png",edge_mask_np2, cmap='gray')

        # 可视化掩码
        # vis_edge_mask(edge_mask1,edge_mask2,edge_mask)
        # vis_edge_mask(None,edge_mask2,None)

        weighted_depth_diff = self.beta * depth_diff * sobel_image_edge + (1 - self.beta) * depth_diff * (1 - edge_mask2)
        focal_loss = torch.mean(weighted_depth_diff) * 0.8


        return focal_loss


class silog_loss_grad(nn.Module):
    def __init__(self, variance):
        super(silog_loss_grad, self).__init__()
        self.focus = variance

    def forward(self, out_edge, depth_edge, mask):
        # 定义Sobel算子的卷积核
        # e_x = torch.abs(out_edge[mask][0] - depth_edge[mask][0])
        # e_y = torch.abs(out_edge[mask][1] - depth_edge[mask][1])
        # mask = grad_mask(mask)
        # 记得看一下这的数值
        out_x = out_edge[0].clone()
        out_y = out_edge[1].clone()
        d_x = depth_edge[0].clone()
        d_y = depth_edge[1].clone()
        out_x[mask == False] = 0
        out_y[mask == False] = 0
        d_x[mask == False] = 0
        d_y[mask == False] = 0
        # print("depth_",depth_edge[0][mask].shape)
        e_x_m_1 = torch.mean(torch.abs(d_x - out_x), dim=(2, 3))
        e_y_m_1 = torch.mean(torch.abs(d_y - out_y), dim=(2, 3))
        # print("shape1",e_x_m_1.shape)
        # print("e_m_1",e_x_m_1,e_y_m_1)
        e_x_m = torch.mean(e_x_m_1)
        e_y_m = torch.mean(e_y_m_1)
        # print("e_m",e_x_m,e_y_m)
        # g_x = torch.log(out_edge[0][mask] + 1e-7) - torch.log(depth_edge[0][mask] + 1e-7)
        # g_y = torch.log(out_edge[1][mask] + 1e-7) - torch.log(depth_edge[1][mask] + 1e-7)
        loss_grad = (torch.log(e_x_m + 0.9) + torch.log(e_y_m + 0.9)) * self.focus
        # print("loss_grad", e_x_m, loss_grad)
        # loss_grad = (torch.var(g) + 0.1 * torch.pow(torch.mean(g), 2)) * self.focus
        return loss_grad


class loss_normal(nn.Module):
    def __init__(self, focus):
        super(loss_normal, self).__init__()
        self.focus = focus

    def forward(self, out_edge, depth_edge, mask):
        # loss: normal
        ones = torch.ones(out_edge[0].size(0), 1, out_edge[0].size(2), out_edge[0].size(3), requires_grad=True).cuda()
        out_x = out_edge[0].clone()
        out_y = out_edge[1].clone()
        d_x = depth_edge[0].clone()
        d_y = depth_edge[1].clone()
        # mask = grad_mask(mask)
        out_x[mask == False] = 0
        out_y[mask == False] = 0
        d_x[mask == False] = 0
        d_y[mask == False] = 0
        # print("shape outx outx",out_x.shape,out_y.shape)
        # print("shape outx outy",d_x.shape,d_y.shape)
        # print("shape outx outy",ones.shape,ones.shape)
        gt_normal = torch.cat((-d_x, -d_y, ones), 1)
        pred_normal = torch.cat((-out_x, -out_y, ones), 1)
        # print("gt_normal type",type(gt_normal))
        # print("gt normal",gt_normal.shape)
        # print("pre normal",pred_normal.shape)
        cos = nn.CosineSimilarity(dim=1)
        loss_normal = torch.mean(torch.abs(1 - cos(pred_normal, gt_normal))) * self.focus
        return loss_normal


class atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(atrous_conv, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True,
                                                                   track_running_stats=True, eps=1.1e-5))

        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels,
                                                                              out_channels=out_channels * 2, bias=False,
                                                                              kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels * 2, momentum=0.01,
                                                                                   affine=True,
                                                                                   track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2,
                                                                              out_channels=out_channels, bias=False,
                                                                              kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation),
                                                                              dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv.forward(x)


class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(upconv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
                              padding=1)
        self.ratio = ratio

    def forward(self, x):
        up_x = torch_nn_func.interpolate(x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.elu(out)
        return out


class reduction_1x1(nn.Sequential):
    def __init__(self, num_in_filters, num_out_filters, max_depth, is_final=False):
        super(reduction_1x1, self).__init__()
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()

        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_module('final',
                                          torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=1, bias=False,
                                                                        kernel_size=1, stride=1, padding=0),
                                                              nn.Sigmoid()))
                else:
                    self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False,
                                                                          kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
                                      torch.nn.Sequential(
                                          nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
                                                    bias=False, kernel_size=1, stride=1, padding=0),
                                          nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2

    def forward(self, net):
        net = self.reduc.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)

        return net


class local_planar_guidance(nn.Module):
    def __init__(self, upratio):
        super(local_planar_guidance, self).__init__()
        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        self.upratio = float(upratio)

    def forward(self, plane_eq, focal):
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]

        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3)).cuda()
        u = (u - (self.upratio - 1) * 0.5) / self.upratio

        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio)).cuda()
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        return n4 / (n1 * u + n2 * v + n3)


class bts(nn.Module):
    def __init__(self, params, feat_out_channels, num_features=512):
        super(bts, self).__init__()
        self.params = params

        self.upconv5 = upconv(feat_out_channels[4], num_features)
        self.bn5 = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)

        self.conv5 = torch.nn.Sequential(
            nn.Conv2d(num_features + feat_out_channels[3], num_features, 3, 1, 1, bias=False),
            nn.ELU())
        self.upconv4 = upconv(num_features, num_features // 2)
        self.bn4 = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4 = torch.nn.Sequential(
            nn.Conv2d(num_features // 2 + feat_out_channels[2], num_features // 2, 3, 1, 1, bias=False),
            nn.ELU())
        self.bn4_2 = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)

        self.daspp_3 = atrous_conv(num_features // 2, num_features // 4, 3, apply_bn_first=False)
        self.daspp_6 = atrous_conv(num_features // 2 + num_features // 4 + feat_out_channels[2], num_features // 4, 6)
        self.daspp_12 = atrous_conv(num_features + feat_out_channels[2], num_features // 4, 12)
        self.daspp_18 = atrous_conv(num_features + num_features // 4 + feat_out_channels[2], num_features // 4, 18)
        self.daspp_24 = atrous_conv(num_features + num_features // 2 + feat_out_channels[2], num_features // 4, 24)
        self.daspp_conv = torch.nn.Sequential(
            nn.Conv2d(num_features + num_features // 2 + num_features // 4, num_features // 4, 3, 1, 1, bias=False),
            nn.ELU())
        self.reduc8x8 = reduction_1x1(num_features // 4, num_features // 4, self.params.max_depth)
        self.lpg8x8 = local_planar_guidance(8)

        self.upconv3 = upconv(num_features // 4, num_features // 4)
        self.bn3 = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(num_features // 4 + feat_out_channels[1] + 1, num_features // 4, 3, 1, 1, bias=False),
            nn.ELU())
        self.reduc4x4 = reduction_1x1(num_features // 4, num_features // 8, self.params.max_depth)
        self.lpg4x4 = local_planar_guidance(4)

        self.upconv2 = upconv(num_features // 4, num_features // 8)
        self.bn2 = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(num_features // 8 + feat_out_channels[0] + 1, num_features // 8, 3, 1, 1, bias=False),
            nn.ELU())

        self.reduc2x2 = reduction_1x1(num_features // 8, num_features // 16, self.params.max_depth)
        self.lpg2x2 = local_planar_guidance(2)

        self.upconv1 = upconv(num_features // 8, num_features // 16)
        self.reduc1x1 = reduction_1x1(num_features // 16, num_features // 32, self.params.max_depth, is_final=True)
        self.conv1 = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 4, num_features // 16, 3, 1, 1, bias=False),
                                         nn.ELU())
        self.get_depth = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False),
                                             nn.Sigmoid())

    def forward(self, features, focal):
        skip0, skip1, skip2, skip3 = features[0], features[1], features[2], features[3]
        dense_features = torch.nn.ReLU()(features[4])
        upconv5 = self.upconv5(dense_features)  # H/16
        upconv5 = self.bn5(upconv5)
        concat5 = torch.cat([upconv5, skip3], dim=1)
        iconv5 = self.conv5(concat5)

        upconv4 = self.upconv4(iconv5)  # H/8
        upconv4 = self.bn4(upconv4)
        concat4 = torch.cat([upconv4, skip2], dim=1)
        iconv4 = self.conv4(concat4)
        iconv4 = self.bn4_2(iconv4)

        daspp_3 = self.daspp_3(iconv4)
        concat4_2 = torch.cat([concat4, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.daspp_12(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat4_4)
        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        daspp_24 = self.daspp_24(concat4_5)
        concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        daspp_feat = self.daspp_conv(concat4_daspp)

        reduc8x8 = self.reduc8x8(daspp_feat)
        plane_normal_8x8 = reduc8x8[:, :3, :, :]
        plane_normal_8x8 = torch_nn_func.normalize(plane_normal_8x8, 2, 1)
        plane_dist_8x8 = reduc8x8[:, 3, :, :]
        plane_eq_8x8 = torch.cat([plane_normal_8x8, plane_dist_8x8.unsqueeze(1)], 1)
        depth_8x8 = self.lpg8x8(plane_eq_8x8, focal)
        depth_8x8_scaled = depth_8x8.unsqueeze(1) / self.params.max_depth
        depth_8x8_scaled_ds = torch_nn_func.interpolate(depth_8x8_scaled, scale_factor=0.25, mode='nearest')

        upconv3 = self.upconv3(daspp_feat)  # H/4
        upconv3 = self.bn3(upconv3)
        concat3 = torch.cat([upconv3, skip1, depth_8x8_scaled_ds], dim=1)
        iconv3 = self.conv3(concat3)

        reduc4x4 = self.reduc4x4(iconv3)
        plane_normal_4x4 = reduc4x4[:, :3, :, :]
        plane_normal_4x4 = torch_nn_func.normalize(plane_normal_4x4, 2, 1)
        plane_dist_4x4 = reduc4x4[:, 3, :, :]
        plane_eq_4x4 = torch.cat([plane_normal_4x4, plane_dist_4x4.unsqueeze(1)], 1)
        depth_4x4 = self.lpg4x4(plane_eq_4x4, focal)
        depth_4x4_scaled = depth_4x4.unsqueeze(1) / self.params.max_depth
        depth_4x4_scaled_ds = torch_nn_func.interpolate(depth_4x4_scaled, scale_factor=0.5, mode='nearest')

        upconv2 = self.upconv2(iconv3)  # H/2
        upconv2 = self.bn2(upconv2)
        concat2 = torch.cat([upconv2, skip0, depth_4x4_scaled_ds], dim=1)
        iconv2 = self.conv2(concat2)

        reduc2x2 = self.reduc2x2(iconv2)
        plane_normal_2x2 = reduc2x2[:, :3, :, :]
        plane_normal_2x2 = torch_nn_func.normalize(plane_normal_2x2, 2, 1)
        plane_dist_2x2 = reduc2x2[:, 3, :, :]
        plane_eq_2x2 = torch.cat([plane_normal_2x2, plane_dist_2x2.unsqueeze(1)], 1)
        depth_2x2 = self.lpg2x2(plane_eq_2x2, focal)
        depth_2x2_scaled = depth_2x2.unsqueeze(1) / self.params.max_depth

        upconv1 = self.upconv1(iconv2)
        reduc1x1 = self.reduc1x1(upconv1)
        concat1 = torch.cat([upconv1, reduc1x1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled], dim=1)
        iconv1 = self.conv1(concat1)
        final_depth = self.params.max_depth * self.get_depth(iconv1)
        if self.params.dataset == 'kitti':
            final_depth = final_depth * focal.view(-1, 1, 1, 1).float() / 715.0873

        return depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, final_depth


class encoder(nn.Module):
    def __init__(self, params):
        super(encoder, self).__init__()
        self.params = params
        import torchvision.models as models
        if params.encoder == 'densenet121_bts':
            self.base_model = models.densenet121(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [64, 64, 128, 256, 1024]
        elif params.encoder == 'densenet161_bts':
            self.base_model = models.densenet161(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [96, 96, 192, 384, 2208]
        elif params.encoder == 'resnet50_bts':
            self.base_model = models.resnet50(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnet101_bts':
            self.base_model = models.resnet101(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnext50_bts':
            self.base_model = models.resnext50_32x4d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnext101_bts':
            self.base_model = models.resnext101_32x8d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'mobilenetv2_bts':
            self.base_model = models.mobilenet_v2(pretrained=True).features
            self.feat_inds = [2, 4, 7, 11, 19]
            self.feat_out_channels = [16, 24, 32, 64, 1280]
            self.feat_names = []
        else:
            print('Not supported encoder: {}'.format(params.encoder))

    def forward(self, x):
        feature = x
        skip_feat = []
        i = 1
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(feature)
            if self.params.encoder == 'mobilenetv2_bts':
                if i == 2 or i == 4 or i == 7 or i == 11 or i == 19:
                    skip_feat.append(feature)
            else:
                if any(x in k for x in self.feat_names):
                    skip_feat.append(feature)
            i = i + 1
        return skip_feat


class BtsModel(nn.Module):
    def __init__(self, params):
        super(BtsModel, self).__init__()
        self.encoder = encoder(params)
        self.decoder = bts(params, self.encoder.feat_out_channels, params.bts_size)

    def forward(self, x, focal):
        skip_feat = self.encoder(x)
        return self.decoder(skip_feat, focal)
