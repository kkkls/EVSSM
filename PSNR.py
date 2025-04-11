from basicsr.metrics.psnr_ssim import calculate_psnr,calculate_ssim
import numpy as np
import math
import cv2
import os

gt_path = '/data0/konglingshun/Test_code_CVPR_2025/EVSSM_model/results_final_2/GoPro_stage_3_2_297/GoPro/'
result_path = '/data0/konglingshun/dataset/Test/GoPro/test/target/'
#result_path = '/data0/konglingshun/dataset/SPA/test/target/'
#result_path = '/data0/konglingshun/dataset/HIDE/test/target/'
psnrs=0
ssims=0
for img in os.listdir(gt_path):
    gt = cv2.imread(gt_path+img)
    result = cv2.imread(result_path+img)
    psnr = calculate_psnr(result,gt,crop_border=0)
    ssim = calculate_ssim(result,gt,crop_border=0)
    psnrs = psnrs+psnr
    ssims = ssims+ssim
    print(img,psnr,ssim)

print(psnrs/len(os.listdir(gt_path)),ssims/len(os.listdir(gt_path)))

