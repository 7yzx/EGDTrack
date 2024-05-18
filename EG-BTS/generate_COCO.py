from __future__ import absolute_import, division, print_function

import os
import argparse
import sys
import torch.nn.functional as F
from tqdm import tqdm
from bts_dataloader import *
from Utiltest import Test_dataset
from bts import *

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
                    default='./models/bts_nyu_v2_pytorch_test/model_rms_gradorin4392')
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


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)


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
    print(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    # 图片所在文件夹路径
    datacoco_dir = '/home/whuai/dataset_track/COCO/train2017/color/train2017'
    showcoco_dir = '/home/whuai/dataset_track/COCO/depth_show_grad08orin'
    save_dir = '/home/whuai/dataset_track/COCO/depth_grad08'
    # data_dir = './coco'

    test_dataset = Test_dataset(root_dir=datacoco_dir)
    test_loader = DataLoader(test_dataset, batch_size=1)
    # 遍历数据加载器
    for i, sample in tqdm(enumerate(test_loader)):
        image = sample['image'].cuda()
        name = sample['name']
        # save_dir = name[0].split('.')[1].split('/')[1]
        # save_name= name[0].split('.')[1].split('/')[2]
        save_name = name[0].split('/')[-1].split('.')[0]
        save_path = os.path.join(save_dir, save_name + '.png')

        focal = 518.8579
        image, pad_way, pad_tb, pad_lr = crop_pad(image)
        lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)

        depth_est = restore_image_size(depth_est, pad_way, pad_tb, pad_lr)
        pred_depth = depth_est.detach().cpu().numpy().squeeze()

        pred_depth_scaled = pred_depth * 1000.0

        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)

        cv2.imwrite(save_path, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if i % 500 ==0:
            vis_path = os.path.join(showcoco_dir, save_name + '_vis.png')
            plt.imsave(vis_path, pred_depth, cmap='magma_r')


def test_fewpic_vis(args):
    """Test function."""
    args.mode = 'test'
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path)
    print(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    # 图片所在文件夹路径
    datacoco_dir = './coco'
    showcoco_dir = './coco_show'
    save_dir = '/home/whuai/dataset_track/COCO/depth_grad08'
    # data_dir = './coco'

    test_dataset = Test_dataset(root_dir=datacoco_dir)
    test_loader = DataLoader(test_dataset, batch_size=1)
    # 遍历数据加载器
    for i, sample in tqdm(enumerate(test_loader)):
        image = sample['image'].cuda()
        name = sample['name']
        # save_dir = name[0].split('.')[1].split('/')[1]
        # save_name= name[0].split('.')[1].split('/')[2]
        save_name = name[0].split('/')[-1].split('.')[0]
        save_path = os.path.join(save_dir, save_name + '.png')

        focal = 518.8579
        image, pad_way, pad_tb, pad_lr = crop_pad(image)
        lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)

        depth_est = restore_image_size(depth_est, pad_way, pad_tb, pad_lr)
        pred_depth = depth_est.detach().cpu().numpy().squeeze()

        pred_depth_scaled = pred_depth * 1000.0

        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)

        cv2.imwrite(save_path, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if i % 500 ==0:
            vis_path = os.path.join(showcoco_dir, save_name + '_vis.png')
            plt.imsave(vis_path, pred_depth, cmap='magma_r')



if __name__ == '__main__':
    # test_fewpic(args)
    test_fewpic_vis(args)