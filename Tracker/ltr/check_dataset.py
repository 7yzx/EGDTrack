import os, sys, glob
from tqdm import tqdm
sequences_list = []
with open('./data_specs/depthtrack_train.txt') as ff:
    for line in ff:
        sequences_list.append(line.rstrip())

num_s = len(sequences_list)
depth_dir = '/home/whuai/dataset_track/depthtrack/train/'
txt_len = 0

for i in tqdm(range(0,num_s)):
    depth_path = os.path.join(depth_dir,sequences_list[i],'color')
    gt_txt = os.path.join(depth_dir,sequences_list[i],'groundtruth.txt')
    jpg_files = glob.glob(os.path.join(depth_path, '*.jpg'))
    jpg_len = len(jpg_files)
    txt_len = 0

    with open(gt_txt, 'r') as file:
        lines = file.readlines()
        txt_len = len(lines)
    if jpg_len != txt_len:
        print("!!!!!!!!")
        print("sequence {}jpglen {}gtlen {}".format(sequences_list[i],jpg_len,txt_len))