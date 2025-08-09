import numpy as np
import glob
import cv2
# mr 512 CT 384
''''
20160222_081031_mask

'''
image = cv2.imread(glob.glob(r'H:\RoadSeg\data\val\image\*.jpg')[3], 0)

target = cv2.imread(glob.glob(r'H:\RoadSeg\data\val\mask\*.png')[3], 0)

a = cv2.imread(glob.glob(r'H:\RoadSeg\trainingrecords\pred_val_TransUnet-Dy-TriAtt\mydata_unet_Lovasz\*.jpg')[3], 0)
b = cv2.imread(glob.glob(r'H:\RoadSeg\trainingrecords\pred_val_TransUnet-TriAtt\mydata_unet_Lovasz\*.jpg')[3], 0)
c = cv2.imread(glob.glob(r'H:\RoadSeg\trainingrecords\pred_val_TransUnet-Dy\mydata_unet_Lovasz\*.jpg')[3], 0)
d = cv2.imread(glob.glob(r'H:\RoadSeg\trainingrecords\pred_val_TransUnet\mydata_unet_Lovasz\*.jpg')[3], 0)
e = cv2.imread(glob.glob(r'H:\RoadSeg\trainingrecords\pred_val_SwinUNet\mydata_unet_Lovasz\*.jpg')[3], 0)
f = cv2.imread(glob.glob(r'H:\RoadSeg\trainingrecords\pred_val_unet\mydata_unet_Lovasz\*.jpg')[3], 0)

new = np.hstack([image, target, a, b, c, d, e, f])
cv2.imwrite(r'H:\RoadSeg\trainingrecords\3.png', new)


