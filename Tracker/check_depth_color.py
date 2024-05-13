import os
from tqdm import tqdm
depth_dir = '/home/whuai/dataset_track/GOT-10K/train/depth'
color_dir = '/home/whuai/dataset_track/GOT-10K/train/color'
seq_name = []
seq_list = []
folders = os.listdir(depth_dir)
for i in tqdm(range(0,len(folders))):
    counts = 0
    color_path = os.path.join(color_dir,folders[i],'groundtruth.txt')
    with open(color_path,'r') as f:
        lines = f.readlines()
        pic_len = len(lines)
    depth_path = os.path.join(depth_dir,folders[i])
    contents = os.listdir(depth_path)
    for item in contents:
        if item.endswith('.png'):
            counts += 1
    if pic_len!=counts:
        print("there is an error in {},color len {},depth len {}".format(folders[i],pic_len,counts))
    seq_name.append(folders[i])
    seq_list.append(int(folders[i].split('_')[-1])-1)

with open('/media/whuai/Windows-SSD/Users/admin/Desktop/my_got10k_train.txt','w') as f:
    for i in range(0,len(seq_list)):
        f.write(str(seq_list[i])+'\n')
print(len(folders))

