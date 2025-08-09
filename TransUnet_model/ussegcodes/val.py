# -*- coding: utf-8 -*-
#!/usr/bin/python3
# -*- coding: utf-8 -*
from random import random

import torch
from torch.autograd import Variable
from models import loss_function, unet, transunet
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from collections import OrderedDict
import cv2
from utils.metrics import compute_metrics

# 权重地址
train_weights = r'H:\USSeg\trainingrecords\checkpoint\mydata_unet_Lovasz\mydata_unet_Lovasz_0.7052_48.pkl'
# 选择网络模型
# 模型声明
model_name = 'UNet'     # 'unet', 'transunet'
net = unet.UNet(in_channels=1, num_classes=3)

ckpt = torch.load(train_weights)
ckpt = ckpt['model_state_dict']
new_state_dict = OrderedDict()
for k, v in ckpt.items():
    new_state_dict[k] = v  # 新字典的key值对应的value为一一对应的值。
net.load_state_dict(new_state_dict)
net.eval()


pre_data_path = r'H:\USSeg\usdata\val\image'

dst_path = '\\'.join(train_weights.split('\\')[:-1]).replace('checkpoint', 'pred_val_' + model_name)
os.makedirs(dst_path, exist_ok=True)


image_list = []
for file in os.listdir(pre_data_path):
    image_list.append(os.path.join(pre_data_path, file))
# 验证
miou_total, mdsc_total, ac_total, mpc_total, mse_total \
    , msp_total, mf1_total = 0, 0, 0, 0, 0, 0, 0
nums = len(image_list)

predictions_all = []
labels_all = []

with torch.no_grad():
    i = 0
    for image in image_list:
        print(i)
        i += 1
        name = image.split("\\")[-1]
        labels = cv2.imread(image.replace('image', 'mask').replace('.jpg', '_mask.png'), 0)

        new_mask = np.zeros((512, 512), dtype=np.uint8)
        new_mask[np.where(labels == 100)] = 1
        new_mask[np.where(labels == 255)] = 2

        org_mask = np.zeros((512, 512), dtype=np.uint8)
        org_mask[np.where(labels == 100)] = 100
        org_mask[np.where(labels == 255)] = 255



        image = Image.open(image).convert('L')
        ori_size = image.size
        image = transforms.ToTensor()(image)
        image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)
        outputs = net(image)
        if isinstance(outputs, list):
            # 若使用deep supervision，用最后一个输出来进行预测
            predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(np.int)
        else:
            # 将概率最大的类别作为预测的类别
            predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int)
        labels = new_mask.astype(np.int)
        predictions_all.append(predictions)
        labels_all.append(labels)
        if isinstance(outputs, list):
            outputs = outputs[-1].squeeze(0)    # [2, 224, 224]
        else:
            outputs = outputs.squeeze(0)    # [2, 224, 224]
        mask = torch.max(outputs, 0)[1].cpu().numpy()
        mask = mask.astype(np.uint8)

        # 下面进行拼接展示,依次为原图，GTmask，预测的mask
        mask[np.where(mask == 1)] = 100
        mask[np.where(mask == 2)] = 255

        new = np.hstack([org_mask, mask])

        cv2.imwrite(dst_path + '\\' + str(name), new)

    # 使用混淆矩阵计算语义分割中的指标
    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(predictions_all, labels_all,
                                                                                   num_classes=3)
    print(
        'Testing: MIoU: {:.4f}, MDSC: {:.4f}, MPC: {:.4f}, AC: {:.4f}, MSE: {:.4f}, MSP: {:.4f}, MF1: {:.4f}'.format(
            miou, mdsc, mpc, ac, mse, msp, mf1
        ))



