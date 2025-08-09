#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import random
import cv2
import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk

# def to255(img):
#     min_val = img.min()
#     max_val = img.max()
#     img = (img - min_val) / (max_val - min_val + 1e-5)  # 图像归一化
#     img = img * 255  # *255
#     return img

def to255(img):
    img = img.astype(np.float32)  # 转换为 float32，降低内存使用
    min_val = img.min()
    max_val = img.max()
    img = (img - min_val) / (max_val - min_val + 1e-5)
    img = img * 255
    return img.astype(np.uint8)  # 输出也可以直接用 uint8，节省空间

save_path = r'D:\TransUnet\training_user_data' # 保存png路径

train_image_path = save_path + '\\' + 'train\\' + 'image'
test_image_path = save_path + '\\' + 'val\\' + 'image'
train_mask_path = save_path + '\\' + 'train\\' + 'mask'
test_mask_path = save_path + '\\' + 'val\\' + 'mask'

os.makedirs(train_image_path, exist_ok=True)
os.makedirs(test_image_path, exist_ok=True)
os.makedirs(train_mask_path, exist_ok=True)
os.makedirs(test_mask_path, exist_ok=True)

names = glob.glob(r"D:\Data_Science_project-data\acouslic-ai-train-set\**\*.mha", recursive=True)
random.shuffle(names)
order = 0

for name in names:
    print(name)

    img = sitk.ReadImage(name)
    img_data = sitk.GetArrayFromImage(img)

    seg = sitk.ReadImage(name.replace('image', 'mask').replace('stacked-fetal-ultrasound', 'stacked_fetal_abdomen'))
    seg_vol = sitk.GetArrayFromImage(seg)
    print(np.unique(seg_vol))

    image_vol = to255(img_data)
    print(seg_vol.shape)
    z = seg_vol.shape[0]
    for i in range(z):
        this_seg = seg_vol[i, :, :]
        if this_seg.max() > 0:
            # 说明有病灶
            this_image = image_vol[i, :, :]
            # 进行resize
            this_image = cv2.resize(this_image, (512, 512), cv2.INTER_LINEAR)
            this_seg = cv2.resize(this_seg, (512, 512), cv2.INTER_NEAREST)
            new_mask = np.zeros((512, 512), dtype=np.uint8)
            new_mask[np.where(this_seg == 1)] = 100
            new_mask[np.where(this_seg == 2)] = 255
            if order % 5 == 0:
                cv2.imwrite(test_mask_path + '\\' + str(order) + '_' + str(i) + '.png', new_mask)
                cv2.imwrite(test_image_path + '\\' + str(order) + '_' + str(i) + '.png', this_image)
            else:
                cv2.imwrite(train_mask_path + '\\' + str(order) + '_' + str(i) + '.png', new_mask)
                cv2.imwrite(train_image_path + '\\' + str(order) + '_' + str(i) + '.png', this_image)
        else:
            pass
    order += 1
    print(order)
