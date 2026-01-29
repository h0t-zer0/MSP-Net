import time

import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.MSPNet import MSPNet
from data_cod import test_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=416, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default='./TestDataset/', help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = MSPNet()
model.load_state_dict(torch.load('./MSPNet_ckpt.pth'))  ## change the path to your saved model
model.cuda()
model.eval()

test_datasets = ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
for dataset in test_datasets:
    save_path = './inference/' + dataset + '/'  ## test dataset predictions save path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/Imgs/'
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    total_time = 0
    count = 0
    for i in tqdm(range(test_loader.size)):
        image, gt, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        start_time = time.perf_counter()
        res, res1, res2 = model(image)
        end_time = time.perf_counter()
        count += 1
        total_time += end_time - start_time
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # print('save img to: ', save_path + name)
        cv2.imwrite(save_path + name, res*255)
    fps = count/total_time
    print('FPS:', fps)
    #print('Inference Done!')
