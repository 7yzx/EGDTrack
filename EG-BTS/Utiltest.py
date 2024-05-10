import os
import time
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets
from PIL import Image
from torchvision import transforms
import csv
import pandas

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def transform_proccess_Lasot():
    return transforms.Compose([
        ToTensor()
    ])


class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image = sample['image']
        if len(image.shape) == 2:
            image = image[..., np.newaxis].repeat([3],axis=2)

        image = self.to_tensor(image)
        image = self.normalize(image)

        return {'image': image}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class Test_dataset(Dataset):
    def __init__(self,root_dir,transform=transform_proccess_Lasot()):
        self.root = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root,os.listdir(self.root)[idx])

        img = np.asarray( Image.open(img_name).convert('RGB'),dtype=np.float32)/255.0

        sample = {'image': img}

        if self.transform:
            sample = self.transform(sample)

        sample['name'] = img_name

        return sample


class Test_datasetwithcsvfile(Dataset):
    def __init__(self,root_dir,file,transform=transform_proccess_Lasot()):
        self.root = root_dir
        self.csv_file = file
        self.transform = transform
        with open(self.csv_file,'r') as f:
            self.lines = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_name = self.lines[idx].split('?')[-2]

        img = np.asarray( Image.open(img_name).convert('RGB'),dtype=np.float32)/255.0

        sample = {'image': img}

        if self.transform:
            sample = self.transform(sample)

        sample['name'] = img_name

        return sample


class Test_got10k_sorted_pic(Dataset):
    def __init__(self,root_dir,sort_split,transform=transform_proccess_Lasot()):
        self.root = root_dir
        self.sequence_list = self._get_sequence_list('list.txt')
        self.file = sort_split
        self.seq_ids = pandas.read_csv(self.file, header=None, squeeze=True, dtype=np.int64).values.tolist()
        print("loading got10k dataset ing......")
        self.transform = transform

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]#读取当前应该读的seq的序号，然后要去和sequence里的对应去
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name, "{:06d}.jpg".format(1))
        img = np.asarray(Image.open(seq_path).convert('RGB'), dtype=np.float32)/255.0

        sample = {'image': img}

        if self.transform:
            sample = self.transform(sample)

        sample['name'] = seq_name
        return sample

    def _get_sequence_list(self,split_txt):
        with open(split_txt, 'r') as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list



class Test_got10kdataset(Dataset):
    def __init__(self,root_dir,split,transform=transform_proccess_Lasot()):
        self.root = root_dir
        self.sequence_list = self._get_sequence_list()
        file_path = split
        seq_ids = pandas.read_csv(file_path, header=None, squeeze=True, dtype=np.int64).values.tolist()

        # self.images = self._make_dataset()
        print("loading got10k dataset ing......")
        sc = time.time()
        load_root_dir = os.path.join(self.root, 'color')
        self.dataset = datasets.ImageFolder(load_root_dir)
        # split train
        self.images = list(filter(lambda x: x[1] in seq_ids, self.dataset.imgs))
        ec = time.time()
        # need_seq_name = [self.dataset.classes[idx] for idx in seq_ids]
        # need_seq_name_path = [os.path.join(load_root_dir, need_seq_name[idx], '00000001.jpg')
        #                       for idx in range(len(need_seq_name))]
        # self.read_first(need_seq_name_path)
        print("load over cost:", round(ec-sc,2))

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx][0]

        img = np.asarray(Image.open(img_path).convert('RGB'), dtype=np.float32)/255.0

        sample = {'image': img}

        if self.transform:
            sample = self.transform(sample)

        sample['name'] = img_path
        return sample

    def read_first(self,list_path):
        seq_namelist = []

        whlist = []
        dict_first_size = {} # 图片的seq_name，path位置，size大小
        got10k_train_3080_split = []
        got10k_train_v100_split = []
        seq_per_size = {}
        for i in range(0, len(list_path)):
            img = Image.open(list_path[i])
            w, h = img.size
            seq_name = list_path[i].split('/')[-2]
            # img.save(os.path.join(self.root,'size_show_every',seq_name+'.jpg'))
            seq_namelist.append(seq_name)
            whlist.append((h, w))

        for idx, wh in enumerate(whlist):
            seq_name_s = seq_namelist[idx]
            seq_name_idx = int(seq_name_s.split('_')[-1]) - 1
            if wh not in dict_first_size:
                dict_first_size[wh] = (os.path.join(self.root,'size_show_every',seq_name_s+'.jpg'),[seq_name_idx])
            else:
                dict_first_size[wh][1].append(seq_name_idx)
            if wh[0]*wh[1]<=720*1200:
                got10k_train_3080_split.append(seq_name_idx)
            else:
                got10k_train_v100_split.append(seq_name_idx)

        sorted_dict = dict(sorted(dict_first_size.items(), key=lambda x: x[0][0] * x[0][1]))
        # whlist = list(dict_first_size.keys())
        with open('my_dict.txt', 'w') as file:
            # 遍历字典中的键值对
            for key, value in sorted_dict.items():
                # 将键值对写入文件，以适当的格式
                file.write(f'{key}?{value[0]}?{value[1]}\n')

        with open("got10k_train_3080_split.txt", 'w') as f1:
            for i in range(0, len(got10k_train_3080_split)):
                f1.write(str(got10k_train_3080_split[i]) + '\n')

        with open("got10k_train_v100_split.txt", 'w') as f2:
            for i in range(0, len(got10k_train_v100_split)):
                f2.write(str(got10k_train_v100_split[i]) + '\n')

        # with open("first.txt", 'w') as ff:
        #     for i in range(0, len(list_path)):
        #         ff.write(seq_namelist[i] + str(list_path[i]) + ' ' + str(hlist[i]) + ' ' + str(wlist[i]) + '\n')


    def _make_dataset(self):
        images = []
        print("loading got-10k ........")
        sc = time.time()
        load_root_dir = os.path.join(self.root,'color')
        self.dataset = datasets.ImageFolder(load_root_dir)

        # for class_name in self.sequence_list:
        #     class_dir = os.path.join(self.root,'color', class_name)
        #     if not os.path.isdir(class_dir):
        #         continue
        #     for img_name in os.listdir(class_dir):
        #         img_path = os.path.join(class_dir, img_name)
        #         if os.path.isfile(img_path):
        #             item = (img_path, self._seqname_to_index(class_name))
        #             images.append(item)

        images = self.dataset.imgs

        ec = time.time()
        print("cost time :", ec-sc)
        return images

    def _seqname_to_index(self,seq_name):
        index_seq = self.sequence_list.index(seq_name)
        return index_seq

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class


