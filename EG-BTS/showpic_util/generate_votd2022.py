from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.autograd import Variable
import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from bts_dataloader import *
from Utiltest import Test_dataset
import misc

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args
# model
parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2_pytorch_test')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load',
                    default='./models/bts_nyu_v2_pytorch_test/model')
# data
parser.add_argument('--data_path', type=str, help='path to the data',
                    default='../../dataset/nyu_depth_v2/official_splits/test/')
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file',
                    default='../train_test_inputs/nyudepthv2_test_files_with_gt.txt')
# paraments
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size', type=int, help='initial num_filters in bts', default=512)

parser.add_argument('--save_name', type=str, help='path to a specific checkpoint to load', default='./result_mask_3987')

parser.add_argument('--save_dir', type=str, default='../../../dataset_track/LaSOT1/LaSOT_depth/')
parser.add_argument('--lasot_path', type=str, default='../../../dataset_track/LaSOT1/LaSOTBenchmark/')
if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)
args.save_lpg = True
save_name = "result_mask_3987"
for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def crop_pad(img):
    '''
    Args:
        img: 所有的图像全部都做填充，并且记录如何填充的，之后必须裁剪取，还是选择在左右上下填充
        img:tensor(1,3,h,w)
    Returns:
        pad_way('h','w'),pad_tb(top,bottom) pad_lr(left,right)
    '''
    pad_way = {}
    pad_tb = {}
    pad_lr = {}
    height, width = img.shape[2:]
    h_scale = height // 32
    w_scale = width // 32
    h_remainder = height % 32
    w_remainder = width % 32
    pad_tb['top'] = 0
    pad_tb['bottom'] = 0
    pad_lr['left'] = 0
    pad_lr['right'] = 0
    # 对于h上
    if h_remainder == 0:
        pad_way['h'] = False
    else:
        new_height = (h_scale + 1) * 32
        top_pad = int((new_height - height) / 2)
        bottom_pad = new_height - top_pad - height
        pad_way['h'] = True
        pad_tb['top'] = top_pad
        pad_tb['bottom'] = bottom_pad

    if w_remainder == 0:  # 不需要填充
        pad_way['w'] = False
    else:
        new_weight = (w_scale + 1) * 32
        left_pad = int((new_weight - width) / 2)
        right_pad = new_weight - left_pad - width
        pad_way['w'] = True
        pad_lr['left'] = left_pad
        pad_lr['right'] = right_pad
    if pad_way['h'] or pad_way['w']:
        image = F.pad(img, (pad_lr['left'], pad_lr['right'], pad_tb['top'], pad_tb['bottom']), value=0)

    else:
        return img, pad_way, pad_tb, pad_lr
    return image, pad_way, pad_tb, pad_lr


def restore_image_size(cropped_img, pad_way, pad_tb, pad_rl):
    """
    Args:
        cropped_img: 裁剪后的图像
        pad_way: 填充方式，包含'h'和'w'
        pad_tb: 上下填充的像素数，字典形式，包含'top'和'bottom'

    Returns:
        原始大小的图像
    """
    restored_img = cropped_img.clone()
    if pad_way['h']:
        # 恢复高度
        restored_img = restored_img[:, :, pad_tb['top']:-pad_tb['bottom'], :]
    if pad_way['w']:
        # 恢复宽度
        restored_img = restored_img[:, :, :, pad_rl['left']:-pad_rl['right']]
    return restored_img


def test_fewpic(args):
    """Test function."""
    args.mode = 'test'

    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    # 图片所在文件夹路径
    data_dir = '../../../dataset_track/vot2022Show/rgb/'
    vot_dir = '../../../dataset_track/vot2022Show/'
    # data_dir = '../../../dataset_track/COCO/train2017/'

    test_dataset = Test_dataset(root_dir=data_dir)
    test_loader = DataLoader(test_dataset, batch_size=1)
    # 遍历数据加载器
    for i, sample in enumerate(test_loader):
        image = sample['image'].cuda()
        name = sample['name']
        save_name = name[0].split('/')[6].split('/')[0]
        save_path = vot_dir + '/depth/' + save_name + '_depth.png'

        focal = 518.8579
        image_c = image.clone()
        image_c, pad_way, pad_tb, pad_lr = crop_pad(image_c)
        lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image_c, focal)

        depth_est = restore_image_size(depth_est, pad_way, pad_tb, pad_lr)

        pred_depth = depth_est.detach().cpu().numpy().squeeze()
        plt.imsave("out_no_color.png",pred_depth,cmap='magma_r')
        # vis_path = './' +'/depth/' + save_name + '_vis.png'
        # pred_depth_vis = misc.color(path=None,value=pred_depth)
        # pred_depth_vis = Image.fromarray(pred_depth_vis)
        nnd1 = misc.color(path='out.png',value=pred_depth)
        nnd = misc.color(path=None,value=pred_depth)
        pred_depth_vis = Image.open('out.png').convert('RGB')
        image_nor = misc.unnormalize_tensor(image).detach().cpu().numpy().squeeze().transpose(1,2,0)
        image_vis =Image.fromarray((image_nor*255).astype(np.uint8))

        # plt.imsave(vis_path, pred_depth, cmap='magma_r')
        w = image_vis.size[0]
        h = image_vis.size[1]
        stacked = Image.new("RGB", (w * 2, h)) # new(w,h)
        stacked.paste(image_vis, (0, 0))
        stacked.paste(pred_depth_vis, (w, 0))
        stacked.save(save_path)

        # pred_depth_scaled = pred_depth * 1000.0

        # pred_depth_scaled = pred_depth_scaled.astype(np.uint16)

        # cv2.imwrite(save_path, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    # test(args)
    test_fewpic(args)