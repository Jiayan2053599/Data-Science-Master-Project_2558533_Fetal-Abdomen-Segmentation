#!/usr/bin/python3
# -*- coding: utf-8 -*
import cv2
from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import numpy as np

def process_binary_mask_tensor(mask):
    """对生成的Tensor类型的mask进行处理"""

    assert isinstance(mask, torch.Tensor), '输入的不是Tensor类型'

    # ToTensor()会自动给数据添加一个channel维度，因为是类别标签，不需要这个维度
    if len(mask.shape) == 3:
        mask = torch.squeeze(mask, dim=0)

    mask = mask.to(torch.uint8)
    mask[mask != 0] = 1

    return mask


class Mydata(Dataset):

    CHANNELS_NUM = 1 # 输入图像通道数（单通道灰度图）
    NUM_CLASSES = 3  # 分割类别数（包括背景0、结构1、结构2）

    def __init__(self, mode, transform=None, target_transform=None, BASE_PATH=""):
        print(mode)
        self.items_image, self.items_mask = make_dataset(mode, BASE_PATH)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.items_image)

    def __str__(self):
        return 'mydata'

    def __getitem__(self, index):
        image_path = self.items_image[index]
        mask_path = self.items_mask[index]

        image = Image.open(image_path).convert('L')
        mask = cv2.imread(mask_path, 0)

        new_mask = np.zeros((512, 512), dtype=np.uint8)
        new_mask[np.where(mask == 100)] = 1
        new_mask[np.where(mask == 255)] = 2

        mask = new_mask.astype(np.uint8)
        mask = torch.from_numpy(mask)
        if self.transform:
            image = self.transform(image)

        return image, mask


def make_dataset(mode, base_path):
    print(mode)

    # assert mode in ['train', 'val']
    #
    # path = os.path.join(base_path, mode)
    image_path = os.path.join(base_path, "image")
    mask_path = os.path.join(base_path, "mask")
    # print(image_path)
    image_list = []
    for file in os.listdir(image_path):
        image_list.append(os.path.join(image_path, file))

    mask_list = []
    for file in os.listdir(mask_path):
        mask_list.append(os.path.join(mask_path, file))

    # print(image_list)
    return image_list, mask_list

