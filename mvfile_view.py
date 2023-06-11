# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import shutil
import torch.backends.cudnn as cudnn
import numpy as np
import math
import torchvision
from torch.autograd import Variable
from model import ft_net
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
from utilsView import load_network
from AugFolder import AugFolder, ViewFolder_mv, ViewAugFolder
from tqdm import tqdm
import pdb

try:
    from apex.fp16_utils import *
except ImportError:  # will be 3.x series
    print(
        'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--ms', default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='../mqveri', type=str, help='./test_data')
parser.add_argument('--name', default='resnet_view_again', type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='save model path')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--inputsize', default=256, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='batchsize')
parser.add_argument('--w', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_NAS', action='store_true', help='use densenet121')
parser.add_argument('--use_SE', action='store_true', help='use densenet121')
parser.add_argument('--use_EF4', action='store_true', help='use densenet121')
parser.add_argument('--use_EF5', action='store_true', help='use densenet121')
parser.add_argument('--use_EF6', action='store_true', help='use densenet121')
parser.add_argument('--use_DSE', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--multi', action='store_true', help='use multiple query')
parser.add_argument('--fp16', action='store_true', help='use fp16.')
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
opt = parser.parse_args()
###load config###

opt.nclasses = 8

# which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True


data_transforms = transforms.Compose([
    transforms.Resize((opt.inputsize , opt.inputsize), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


data_dir = test_dir
image_datasets = {}
image_datasets['test'] = ViewFolder_mv(os.path.join(data_dir, 'query_miss_qianhou'),
                                        data_transforms)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=0, pin_memory=True)
                   # 8 workers may work faster
                   for x in ['test']}
use_gpu = torch.cuda.is_available()
dataset_sizes = len(image_datasets['test'])
######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extract_feature(model, dataloaders):
    for data in tqdm(dataloaders):  # 读取信息的进度条
        inputs, labels, filepath = data  # 一个batch的data
        #pdb.set_trace()
        #labels = labels.long()
        inputs = Variable(inputs.cuda().detach())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        #pdb.set_trace()
        if preds != 2:
            dst = '../mqveri/query_miss_qianhou/' + filepath[0].split('/')[-2]
            #shutil.copy(filepath[0], os.path.join(dst, filepath[0].split('/')[-1]))
            os.remove(os.path.join(dst, filepath[0].split('/')[-1]))
        '''
        elif preds == 1:
            dst = '../mqveri/hou/' + filepath[0].split('/')[-2]
            if not os.path.isdir(dst):
                os.mkdir(dst)
            shutil.copy(filepath[0], os.path.join(dst, filepath[0].split('/')[-1]))
        elif preds == 2:
            dst = '../mqveri/ce/' + filepath[0].split('/')[-2]
            if not os.path.isdir(dst):
                os.mkdir(dst)
            shutil.copy(filepath[0], os.path.join(dst, filepath[0].split('/')[-1]))
        '''

    return labels
######################################################################
# Load Collected data Trained model
print('-------test-----------')
model = ft_net(3, droprate=0, stride = 1, ibn = False)
view_path = os.path.join('../mqveri/outputs', "viewagain", "net_last.pth")
# Change to test mode
#model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(view_path))
model = model.cuda()
#model = model.module
model = model.eval()
# Extract feature
with torch.no_grad():
    acc = extract_feature(model, dataloaders['test'])
    print(acc)
