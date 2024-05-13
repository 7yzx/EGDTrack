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
from Utiltest import Test_dataset,Test_got10kdataset,Test_datasetwithcsvfile,Test_got10k_sorted_pic


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
                    default='./models/bts_nyu_v2_pytorch_test/model_rms_4371')
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


def test(params):
    """Test function."""
    args.mode = 'test'
    dataloader = BtsDataLoader(args, 'test')

    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_test_samples = get_num_lines(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    pred_depths = []

    start_time = time.time()
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())

            image = crop_pad(image)

            # Predict
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            pred_depths.append(depth_est.cpu().numpy().squeeze())

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')

    print('Saving result pngs..')
    if not os.path.exists(os.path.dirname(save_name)):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/raw')
            os.mkdir(save_name + '/cmap')
            os.mkdir(save_name + '/rgb')
            os.mkdir(save_name + '/gt')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    for s in tqdm(range(num_test_samples)):

        scene_name = lines[s].split()[0].split('/')[0]
        filename_pred_png = save_name + '/raw/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
            '.jpg', '.png')
        filename_cmap_png = save_name + '/cmap/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
            '.jpg', '.png')
        filename_gt_png = save_name + '/gt/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
            '.jpg', '.png')
        filename_image_png = save_name + '/rgb/' + scene_name + '_' + lines[s].split()[0].split('/')[1]

        rgb_path = os.path.join(args.data_path, './' + lines[s].split()[0])
        image = cv2.imread(rgb_path)
        if args.dataset == 'nyu':
            gt_path = os.path.join(args.data_path, './' + lines[s].split()[1])
            gt = cv2.imread(gt_path, -1).astype(np.float32) / 1000.0  # Visualization purpose only
            gt[gt == 0] = np.amax(gt)

        pred_depth = pred_depths[s]

        pred_depth_scaled = pred_depth * 1000.0

        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        cv2.imwrite(filename_image_png, image[10:-1 - 9, 10:-1 - 9, :])
        plt.imsave(filename_gt_png, np.log10(gt[10:-1 - 9, 10:-1 - 9]), cmap='Greys')
        pred_depth_cropped = pred_depth[10:-1 - 9, 10:-1 - 9]
        plt.imsave(filename_cmap_png, np.log10(pred_depth_cropped), cmap='Greys')

    return


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
    print("load checkpoint",args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    data_dir = '/home/whuai/dataset_track/GOT-10K/train/'
    test_dataset = Test_got10kdataset(root_dir=data_dir,split='./split_data/got10k_sorted_train3080_list.txt')
    test_loader = DataLoader(test_dataset, batch_size=1)

    with open("./split_data/got10k_sorted_train3080sample_file.txt",'r') as f:
        got10k_sample_list = [line.rstrip() for line in f.readlines()]
    # 遍历数据加载器
    for i, sample in tqdm(enumerate(test_loader)):
        image = sample['image'].cuda()
        img_name = sample['name']
        save_name = img_name[0].split('/')[-1].split('.')[0]
        seq_name = img_name[0].split('/')[-2]
        save_dir = '/home/whuai/dataset_track/GOT-10K/train/depth/' + seq_name
        focal = 518.8579

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir , save_name + '.png')
        if seq_name in got10k_sample_list:
            h = image.shape[2]
            w = image.shape[3]
            image = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=False)
            image, pad_way, pad_tb, pad_lr = crop_pad(image)
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            del lpg2x2,lpg4x4,lpg8x8
            depth_est = restore_image_size(depth_est, pad_way, pad_tb, pad_lr)
            depth_est = F.interpolate(depth_est, size=[h,w], mode='bilinear', align_corners=False)
        else:
            image, pad_way, pad_tb, pad_lr = crop_pad(image)
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            depth_est = restore_image_size(depth_est, pad_way, pad_tb, pad_lr)

        pred_depth = depth_est.detach().cpu().numpy().squeeze()

        pred_depth_scaled = pred_depth * 1000.0

        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)

        cv2.imwrite(save_path, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if i % 1000 == 0:
            vis_dir = '/home/whuai/dataset_track/GOT-10K/train/showgot10k/' + seq_name
            if not os.path.exists(vis_dir):
                os.mkdir(vis_dir)
            vis_path = os.path.join(vis_dir, save_name + '_vis.png')
            plt.imsave(vis_path, pred_depth, cmap='magma_r')
        # h = image.shape[2]
        # w = image.shape[3]
        # image = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=False)

        # image, pad_way, pad_tb, pad_lr = crop_pad(image)
        #
        # lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
        # del lpg2x2,lpg4x4,lpg8x8
        # depth_est = restore_image_size(depth_est, pad_way, pad_tb, pad_lr)
        #
        # depth_est = F.interpolate(depth_est, size=[h,w], mode='bilinear', align_corners=False)
        #
        # pred_depth = depth_est.detach().cpu().numpy().squeeze()


def test_fewpicv100(args):
    """Test function."""
    args.mode = 'test'

    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path)
    print("load checkpoint",args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))
    with open("got10k_v100_file.txt",'r') as f:
        got10k_sample_list = [line.rstrip() for line in f.readlines()]

    # data_dir = '/home/yezxiao/scratch/dataset_track/split_got10k/'
    data_dir = '/home/whuai/dataset_track/GOT-10K/train'
    test_dataset = Test_got10kdataset(root_dir=data_dir,split='got10k_train_v100_split.txt')
    test_loader = DataLoader(test_dataset, batch_size=1)
    # 遍历数据加载器
    for i, sample in tqdm(enumerate(test_loader)):
        image = sample['image'].cuda()
        img_name = sample['name']
        save_name = img_name[0].split('/')[-1].split('.')[0]
        seq_name = img_name[0].split('/')[-2]#GOT-10k_Train_003785
        save_dir = '/home/yezixiao/scratch/dataset_track/split_got10k/depth/' + seq_name
        focal = 518.8579
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir , save_name + '.png')
        if seq_name in got10k_sample_list:
            h = image.shape[2]
            w = image.shape[3]
            image = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=False)
            image, pad_way, pad_tb, pad_lr = crop_pad(image)
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            del lpg2x2,lpg4x4,lpg8x8
            depth_est = restore_image_size(depth_est, pad_way, pad_tb, pad_lr)
            depth_est = F.interpolate(depth_est, size=[h,w], mode='bilinear', align_corners=False)

        else:
            image, pad_way, pad_tb, pad_lr = crop_pad(image)
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            depth_est = restore_image_size(depth_est, pad_way, pad_tb, pad_lr)

        pred_depth = depth_est.detach().cpu().numpy().squeeze()

        pred_depth_scaled = pred_depth * 1000.0

        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)

        cv2.imwrite(save_path, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if i % 1000 == 0:
            vis_dir = '/home/whuai/dataset_track/GOT-10K/train/showgot10k/' + seq_name
            if not os.path.exists(vis_dir):
                os.mkdir(vis_dir)
            vis_path = os.path.join(vis_dir, save_name + '_vis.png')
            plt.imsave(vis_path, pred_depth, cmap='magma_r')




def test_pic_on3090teset(args):
    '''
    2820 处可以操作，86，40000
    '''
    print("test_pic_on3090 for some test ")
    """Test function."""
    args.mode = 'test'
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint_path)
    print("load checkpoint",args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    root_dir = '/home/whuai/dataset_track/GOT-10K/train/size_show_every'
    dataset_test =Test_datasetwithcsvfile(root_dir= root_dir,file='my_dict.txt')
    # dataset_test =Test_datasetwithcsvfile(root_dir= root_dir,file='test_dict.txt') # 测试较大图片的性能

    test_loader = DataLoader(dataset_test, batch_size=1)
    # 遍历数据加载器
    for i, sample in enumerate(test_loader):
        image = sample['image'].cuda()
        image1 = image.clone()
        img_name = sample['name']
        save_name = img_name[0].split('/')[-1].split('.')[0]
        seq_name = img_name[0].split('/')[-2]
        save_dir = './' + 'depth_10k_3090_test_unsample/'
        print("save dir ",save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir , save_name + '.png')
        focal = 518.8579

        image, pad_way, pad_tb, pad_lr = crop_pad(image)
        lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
        depth_est = restore_image_size(depth_est, pad_way, pad_tb, pad_lr)

        pred_depth = depth_est.detach().cpu().numpy().squeeze()
        vis_path = os.path.join(save_dir,str(i)+ save_name + '_vis.png' )
        plt.imsave(vis_path, pred_depth, cmap='magma_r')

        # h = image1.shape[2]
        # w = image1.shape[3]
        # image1 = F.interpolate(image1, scale_factor=0.5, mode='bilinear', align_corners=False)
        # image1, pad_way, pad_tb, pad_lr = crop_pad(image1)
        # lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est1 = model(image1, focal)
        # del lpg2x2, lpg4x4, lpg8x8
        # depth_est1 = restore_image_size(depth_est1, pad_way, pad_tb, pad_lr)
        # depth_est1 = F.interpolate(depth_est1, size=[h, w], mode='bilinear', align_corners=False)
        # pred_depth1 = depth_est1.detach().cpu().numpy().squeeze()
        # vis_path = os.path.join(save_dir ,str(i)+save_name + '_viss.png')
        # plt.imsave(vis_path, pred_depth1, cmap='magma_r')
        print("over ",save_name)
        #
        # pred_depth_scaled = pred_depth * 1000.0
        #
        # pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        #
        # cv2.imwrite(save_path, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def test_pic_on3090_sorted_sample(args):
    '''
    测试全部的图片并且显示,这里是全部resize后处理
    '''
    print("test_pic_on3090 for all pic ")
    """Test function."""
    args.mode = 'test'
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint_path)
    print("load checkpoint",args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    root_dir = '/home/whuai/dataset_track/GOT-10K/train/size_show_every'
    dataset_test =Test_got10k_sorted_pic(root_dir=root_dir,sort_split='got10k_sorted_train_list.txt')
    test_loader = DataLoader(dataset_test, batch_size=1)
    # 遍历数据加载器
    for i, sample in enumerate(test_loader):
        image = sample['image'].cuda()
        image1 = image.clone()
        img_name = sample['name']
        save_name = img_name[0].split('/')[-1].split('.')[0]
        # seq_name = img_name[0].split('/')[-2]
        save_dir = './' + 'depth_10k_3090_every_unsample/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        focal = 518.8579
        h = image1.shape[2]
        w = image1.shape[3]
        image1 = F.interpolate(image1, scale_factor=0.5, mode='bilinear', align_corners=False)
        image1, pad_way, pad_tb, pad_lr = crop_pad(image1)
        lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est1 = model(image1, focal)
        del lpg2x2, lpg4x4, lpg8x8
        depth_est1 = restore_image_size(depth_est1, pad_way, pad_tb, pad_lr)
        depth_est1 = F.interpolate(depth_est1, size=[h, w], mode='bilinear', align_corners=False)
        pred_depth1 = depth_est1.detach().cpu().numpy().squeeze()
        vis_path = os.path.join(save_dir ,str(i)+save_name + '_viss.png')
        plt.imsave(vis_path, pred_depth1, cmap='magma_r')
        print("over ",save_name)
        #
        # pred_depth_scaled = pred_depth * 1000.0
        #
        # pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        #
        # cv2.imwrite(save_path, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    # test(args)
    # test_fewpicv100(args)
    test_fewpic(args)
    # test_pic_on3090teset(args)
    # test_pic_on3090_sorted_sample(args)