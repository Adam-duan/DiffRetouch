import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import json
import random
from ldm.util import instantiate_from_config_vq_diffusion
import albumentations
from torchvision import transforms as trans
import re

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def natural_sort_key(s):
    """
    按文件名的结构排序，即依次比较文件名的非数字和数字部分
    """
    # 将字符串按照数字和非数字部分分割，返回分割后的子串列表
    sub_strings = re.split(r'(\d+)', s)
    # 如果当前子串由数字组成，则将它转换为整数；否则返回原始子串
    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    # 根据分割后的子串列表以及上述函数的返回值，创建一个新的列表
    # 按照数字部分从小到大排序，然后按照非数字部分的字典序排序
    return sub_strings

class PPR10KDiffDataset(Dataset):

    """
    This Dataset can be used for:
    - image-only: setting 'conditions' = []
    - image and multi-modal 'conditions': setting conditions as the list of modalities you need

    To toggle between 256 and 512 image resolution, simply change the 'image_folder'
    """

    def __init__(self,
        phase = 'train',
        test_dataset_size=2286,
        conditions = ['text', 'image'],
        gt_path = '/mnt/lustre/duanzhengpeng/PPR10K/GT',
        gt_folder_list = ['target_a_512', 'target_b_512', 'target_c_512'],
        image_path = '/mnt/lustre/duanzhengpeng/PPR10K/RAW',
        text_folder = '/mnt/lustre/duanzhengpeng/PPR10K/GT/json/',
        drop_ratio = 0
        ):

        self.conditions = conditions

        self.gt_folder_list = gt_folder_list
        print(f'self.gt_folder_list = {self.gt_folder_list}')

        self.image_name_list = sorted(os.listdir(image_path), key=natural_sort_key)

        # train test split
        if phase == 'train':
            self.image_name_list = self.image_name_list[:-test_dataset_size]
        elif phase == 'test':
            self.image_name_list = self.image_name_list[-test_dataset_size:]
        else:
            raise NotImplementedError
        
        if 'text' in self.conditions:
            self.text_folder = text_folder
            print(f'self.text_folder = {self.text_folder}')

        if 'image' in self.conditions:
            self.image_path = image_path
            print(f'self.image_path = {self.image_path}')

        self.gt_path = []
        self.image_path = []
        self.label_list = []

        for gt_folder in gt_folder_list:
            self.gt_path.extend([os.path.join(gt_path, gt_folder, image_name) for image_name in self.image_name_list])
            self.image_path.extend([os.path.join(image_path, image_name) for image_name in self.image_name_list])
            with open(os.path.join(self.text_folder, gt_folder+'.json'), 'r') as f:
                text_file_content = json.load(f)
            for image_name in self.image_name_list:
                self.label_list.append(text_file_content[image_name])
        
        self.num = len(self.gt_path)
        self.drop_ratio = drop_ratio

        # verbose
        print(f'phase = {phase}')
        print(f'number of samples = {self.num}')


    def __len__(self):
        return self.num

    def __getitem__(self, index):

        # ---------- (1) get gt ----------
        gt_path = self.gt_path[index]
        gt = load_img(gt_path)
        gt = np.array(gt).astype(np.uint8)
        gt = gt.astype(np.float32)/127.5 - 1.0

        # record into data entry
        data = {'gt': gt}

        # ---------- (2) get image ----------
        image_path = self.image_path[index]
        image = load_img(image_path)
        image = np.array(image).astype(np.uint8)
        image = image.astype(np.float32)/127.5 - 1.0

        data['image'] = image

        # ---------- (3) get text ----------
        if 'text' in self.conditions:
            label_list = self.label_list[index]
            text = ''
            random.shuffle(label_list)
            for label in label_list:
                if random.random() >= self.drop_ratio:
                    text += label + ', '
            text = text[:-2]
            # record into data entry
            data['txt'] = text

        return data
