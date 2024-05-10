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

from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from bts_dataloader import *

'''
python bts_test_debug.py --checkpoint_path './models/bts_nyu_v2_pytorch_test/model_rms_gradorin4392' --save_name 'grad08'
'''
def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2_pytorch_test')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
# parser.add_argument('--data_path', type=str, help='path to the data', default='../../dataset/nyu_depth_v2/official_splits/test/')

parser.add_argument('--data_path', type=str, help='path to the data', default='../../dataset/')
# parser.add_argument('--filenames_file', type=str, help='path to the filenames text file',default='../train_test_inputs/nyudepthv2_test_files_with_gt.txt')

parser.add_argument('--filenames_file', type=str, help='path to the filenames text file',default='./nyu2_test.txt')
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='./models/bts_nyu_v2_pytorch_test/model_rms_gradonly_43631')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--save_name',type=str)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)
args.save_lpg = True

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
    pred_8x8s = []
    pred_4x4s = []
    pred_2x2s = []
    pred_1x1s = []

    start_time = time.time()
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())
            # Predict
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            pred_depths.append(depth_est.cpu().numpy().squeeze())
            pred_8x8s.append(lpg8x8[0].cpu().numpy().squeeze())
            pred_4x4s.append(lpg4x4[0].cpu().numpy().squeeze())
            pred_2x2s.append(lpg2x2[0].cpu().numpy().squeeze())
            pred_1x1s.append(reduc1x1[0].cpu().numpy().squeeze())

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')
    
    save_name = 'result_' + args.save_name
    
    print('Saving result pngs..')
    if not os.path.exists(os.path.dirname(save_name)):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/raw')
            os.mkdir(save_name + '/cmap')
            os.mkdir(save_name + '/rgb')
            os.mkdir(save_name + '/gt')
            os.mkdir(save_name + '/cmapone')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    for s in tqdm(range(num_test_samples)):
        scene_name = lines[s]
        filename_pred_png = save_name + '/raw/'  + 'pred_' + lines[s].split()[0].split('/')[2].split('_')[0]+'.png'
        filename_cmap_png = save_name + '/cmap/' + 'pred_' + lines[s].split()[0].split('/')[2].split('_')[0]+'.png'
        filename_gt_png = save_name + '/gt/' + 'pred_' + lines[s].split()[0].split('/')[2].split('_')[0]+'.png'
        # filename_cmapone_png = save_name + '/cmapone/' + 'pred_' + lines[s].split()[0].split('/')[2].split('_')[0]+'.png'
        filename_image_png = save_name + '/rgb/' + 'pred_' + lines[s].split()[0].split('/')[2].split('_')[0]+'.png'

        rgb_path = os.path.join(args.data_path , lines[s].split()[0])
        image = cv2.imread(rgb_path)
        if args.dataset == 'nyu':
            gt_path = os.path.join(args.data_path,  lines[s].split()[1])
            gt = cv2.imread(gt_path, -1).astype(np.float32) / 1000.0  # Visualization purpose only
            gt[gt == 0] = np.amax(gt)
        
        pred_depth = pred_depths[s]
        pred_8x8 = pred_8x8s[s]
        pred_4x4 = pred_4x4s[s]
        pred_2x2 = pred_2x2s[s]
        pred_1x1 = pred_1x1s[s]
        
        if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark':
            pred_depth_scaled = pred_depth * 256.0
        else:
            pred_depth_scaled = pred_depth * 1000.0
        
        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        pred_depth_cropped = pred_depth[10:-1 - 9, 10:-1 - 9]
        plt.imsave(filename_cmap_png, np.log10(pred_depth_cropped), cmap='magma_r')

        args.save_lpg = False
        if args.save_lpg:
            cv2.imwrite(filename_image_png, image[10:-1 - 9, 10:-1 - 9, :])
            if args.dataset == 'nyu':
                plt.imsave(filename_gt_png, np.log10(gt[10:-1 - 9, 10:-1 - 9]), cmap='magma_r')
                pred_depth_cropped = pred_depth[10:-1 - 9, 10:-1 - 9]
                plt.imsave(filename_cmap_png, np.log10(pred_depth_cropped), cmap='magma_r')
                pred_8x8_cropped = pred_8x8[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8_cropped), cmap='magma_r')
                pred_4x4_cropped = pred_4x4[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4_cropped), cmap='magma_r')
                pred_2x2_cropped = pred_2x2[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2_cropped), cmap='magma_r')
                pred_1x1_cropped = pred_1x1[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1_cropped), cmap='magma_r')
    
    return


if __name__ == '__main__':
    test(args)
