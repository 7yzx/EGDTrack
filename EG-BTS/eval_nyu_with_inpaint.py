

from __future__ import absolute_import, division, print_function
import os
import argparse
import fnmatch
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from misc import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
'''
For eval with inpaint nyu2depth test dataset and eval with image_generate_gt test
python eval_nyu_with_inpaint.py --pred_path './result_mask_392/raw/' --is_crop '1'
python eval_nyu_with_inpaint.py --pred_path './result_mask_3987/raw/' --is_crop '1'
python eval_nyu_with_inpaint.py --pred_path './result_mask_395/raw/' --is_crop '1'
python eval_nyu_with_inpaint.py --pred_path './result_gradonly/raw/' --is_crop '1'

python eval_nyu_with_inpaint.py --pred_path './result_mask_392/raw/' --is_crop '2'
python eval_nyu_with_inpaint.py --pred_path './result_mask_3987/raw/' --is_crop '2'
python eval_nyu_with_inpaint.py --pred_path './result_mask_395/raw/' --is_crop '2'
'''
def convert_arg_line_to_args(arg_line):
    '''
    Args:
        arg_line: 传入的参数行
    这个函数的作用是将传入的参数行（arg_line）按空格分割，然后逐个返回非空的参数。
    如果参数行中含有多个连续的空格，这个函数会将它们忽略。
    '''
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

# 传入命令行参数
parser = argparse.ArgumentParser(description='BTS TensorFlow implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--pred_path', type=str, help='path to the prediction results in png', default='./zoe_result/')
parser.add_argument('--gt_path', type=str, help='root path to the groundtruth data', default='../../dataset/data/nyu2_test')
parser.add_argument('--dataset', type=str, help='dataset to test on, nyu or kitti', default='nyu')
parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--mode_t', type=str, help='rgb or none to eval with rgb edge and depth gt edge', default='None')
parser.add_argument('--is_crop', type=str, help='1:no 2:crop 3:for rgb, 4:dowmsample' , default='4')

args = parser.parse_args()
args.eigen_crop = True


def test(args):
    if args.mode_t == 'rgb':
        print("This eval is using RGB image to gt")
    else:
        print("This eval is using inpaint depth of gt!!!")

    if args.is_crop == '1' or args.is_crop is None:
        print("use no crop")
    elif args.is_crop == '2':
        print("use crop")
    elif args.is_crop == '3':
        print("rbg")
    elif args.is_crop == '4':
        print("resize 1/4")
    # 初始化存放gt的数组还有数组名字，以及丢失的id
    global gt_depths, missing_ids, pred_filenames,gt_image_gray
    gt_depths = []
    gt_image_gray = []
    missing_ids = set()
    pred_filenames = []
    # 把preddepth的图片名字全部存放在了pred_filenames
    for root, dirnames, filenames in os.walk(args.pred_path):  # 使用os.walk遍历args.pred_path路径下的所有文件和目录
        for pred_filename in fnmatch.filter(filenames, '*.png'):
            if 'cmap' in pred_filename or 'gt' in pred_filename:  # 如果文件名中包含'cmap'或'gt'，则跳过当前文件
                continue
            dirname = root.replace(args.pred_path, '')
            pred_filenames.append(os.path.join(dirname, pred_filename))

    num_test_samples = len(pred_filenames)

    pred_depths = []
    # 预测深度图像存储在pred_depths
    for i in range(num_test_samples):
        pred_depth_path = os.path.join(args.pred_path, pred_filenames[i])  # 文件相对于此Python文件的位置
        pred_depth = cv2.imread(pred_depth_path, -1)
        if pred_depth is None:
            print('Missing: %s ' % pred_depth_path)
            missing_ids.add(i)
            continue
        # 使用的16-bit存储 除1000（一般操作）
        if args.dataset == 'nyu':
            pred_depth = pred_depth.astype(np.float32) / 1000.0
        else:
            pred_depth = pred_depth.astype(np.float32) / 256.0

        pred_depths.append(pred_depth)

    print('Raw png files reading done')
    print('Evaluating {} files'.format(len(pred_depths)))

    # 去找gt深度图，也是16-bit保存，所以、1000
    for t_id in range(num_test_samples):
        file_name = pred_filenames[t_id].split('.')[0].split('_')[-1]
        # file_name = pred_filenames[t_id].split('.')[0].split('_')[0]

        gt_depth_path = os.path.join(args.gt_path, file_name + '_depth.png')
        depth = cv2.imread(gt_depth_path, -1)
        if depth is None:
            print('Missing: %s ' % gt_depth_path)
            missing_ids.add(t_id)
            continue
        depth = depth.astype(np.float32) / 1000.0
        gt_depths.append(depth)
        # reading gt depth file!over
        if args.mode_t == 'rgb':
            # get gt from rgb image
            file_name = pred_filenames[t_id].split('.')[0].split('_')[-1]
            gt_color_path = os.path.join(args.gt_path, file_name + '_colors.png')
            gt_color = cv2.imread(gt_color_path, -1)
            gt_color_gray = (cv2.cvtColor(gt_color,cv2.COLOR_BGR2GRAY) / 255.0).astype(np.float32)

            if gt_color is None:
                print('Missing: %s ' % gt_color_path)
                missing_ids.add(t_id)
                continue

            gt_image_gray.append(gt_color_gray)



    print('GT files reading done')
    print('{} GT files missing'.format(len(missing_ids)))

    print('Computing errors')
    neweval(pred_depths,args)
    print('Done.')

def neweval(pred_depths,args):
    num_samples = len(pred_depths)
    pred_depths_valid = []

    i = 0
    for t_id in range(num_samples):
        if t_id in missing_ids:
            continue

        pred_depths_valid.append(pred_depths[t_id])

    num_samples = num_samples - len(missing_ids)

    # 初始化
    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1 = np.zeros(num_samples, np.float32)
    d2 = np.zeros(num_samples, np.float32)
    d3 = np.zeros(num_samples, np.float32)
    P = np.zeros(num_samples,np.float32)
    R = np.zeros(num_samples,np.float32)
    F = np.zeros(num_samples,np.float32)
    # 正式开始评估
    for i in tqdm(range(num_samples)):
        gt_depth = gt_depths[i]        # gt
        pred_depth = pred_depths_valid[i]  #pred
        if args.mode_t == 'rgb':
            gt_image = gt_image_gray[i]
        # 归一在0-10之间，对于pred
        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        # gt 填充inf和nan
        gt_depth[np.isinf(gt_depth)] = 0
        gt_depth[np.isnan(gt_depth)] = 0
        # 对于gt有效的区域也就是在0-10之间的数据，valid_mask
        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        eval_mask = np.zeros(valid_mask.shape)

        eval_mask[45:471, 41:601] = 1  # 不要边缘的值，可以

        valid_mask = np.logical_and(valid_mask, eval_mask)

        silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(
            gt_depth[valid_mask], pred_depth[valid_mask])
        # 对于评估的方法有三种，
        # 1. 对于修复后（inpaint）的depth作为gt，因为这个样子可以基本看出原来物体的边缘特性，同时和之前的也可以对比
        # 2. 一个问题是inpaint，之前论文中的图像大小是224，224大小，图像大小对于评估的精度有很大影响，图像小，其边缘信息越不明显。
        # 3. 如果之后对于没有gt的深度图如何进行粗略的评估，我觉得可用将image的边缘信息作为gt来评估
        if args.mode_t == 'rgb':
            P[i], R[i], F[i] = compute_edges(gt_image, pred_depth, eval_mask, mode_e=3)
        else:
            P[i], R[i], F[i] = compute_edges(gt_depth, pred_depth, eval_mask,mode_e=args.is_crop)
            # P[i], R[i], F[i] = compute_edges_ok(gt_depth, pred_depth, valid_mask)

    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10'))
    print("{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
        d1.mean(), d2.mean(), d3.mean(),
        abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), silog.mean(), log10.mean()))
    print("{:>7}, {:>7}, {:>7}".format(
        'Pv', 'Rv', 'Fv'))
    print("{:7.3f}, {:7.3f}, {:7.3f}".format(
        P.mean(), R.mean(), F.mean(),
        ))

    return silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3


if __name__ == '__main__':
    # mode = 'rgb'
    print("now is use the ",args.pred_path.split('.')[1].split('/')[1])
    test(args)



