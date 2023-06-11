# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import math
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import torchvision.utils as vutils
from utils_gan import load_network
from AugFolder import AugFolder, ViewFolder, ViewAugFolder
from tqdm import tqdm
import pdb
from model import ft_net, ft_net_angle, ft_net_EF4, ft_net_EF5, ft_net_EF6, ft_net_arc, ft_net_IR, ft_net_SE, \
    ft_net_DSE, ft_net_dense, ft_net_NAS, PCB, PCB_test, CPB
from evaluate_gpu import calculate_result
from evaluate_rerank import calculate_result_rerank

# fp16
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
parser.add_argument('--test_dir', default='../mqveri_gan', type=str, help='./test_data')
parser.add_argument('--name', default='demo', type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='save model path')
parser.add_argument('--batchsize', default=2, type=int, help='batchsize')
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
parser.add_argument('--gan', action='store_true', help='use gan')
parser.add_argument('--multi', action='store_true', help='use multiple query')
parser.add_argument('--fp16', action='store_true', help='use fp16.')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('../mqveri/outputs', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
opt.fp16 = config['fp16']
opt.PCB = config['PCB']
opt.CPB = config['CPB']
opt.inputsize = config['inputsize']
opt.stride = config['stride']
opt.angle = config['angle']
opt.use_EF4 = config['use_EF4']
opt.use_EF5 = config['use_EF5']
opt.use_EF6 = config['use_EF6']

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']

if 'pool' in config:
    opt.pool = config['pool']

opt.use_dense = config['use_dense']
if 'use_NAS' in config:  # compatible with early config
    opt.use_NAS = config['use_NAS']
else:
    opt.use_NAS = False

if 'use_SE' in config:  # compatible with early config
    opt.use_SE = config['use_SE']
else:
    opt.use_SE = False

if 'use_DSE' in config:  # compatible with early config
    opt.use_DSE = config['use_DSE']
else:
    opt.use_DSE = False

if 'use_IR' in config:  # compatible with early config
    opt.use_IR = config['use_IR']
else:
    opt.use_IR = False

if 'arc' in config:
    opt.arc = config['arc']
else:
    opt.arc = False

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

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
SetRange = transforms.Lambda(lambda X: 2 * X - 1.)

data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    #transforms.CenterCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #SetRange

])
'''
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
    SetRange
    transforms.Resize((round(opt.inputsize), round(opt.inputsize)), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    '''

data_dir = test_dir
image_datasets = {}
image_datasets['test'] = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                        data_transforms)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=8, pin_memory=True)
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
    running_corrects = 0.0
    for data in tqdm(dataloaders):  # 读取信息的进度条
        inputs, labels = data  # 一个batch的data
        label = int(labels[0])
        pdb.set_trace()
        outputs = model(inputs)
        #outputs[0] = outputs[0]
        vutils.save_image(outputs[0],
                          f"./gan0000{label}"f".png",
                          normalize=True,
                          nrow=3)
        vutils.save_image(outputs[1],
                          f"./ganori{label}"f".png",
                          normalize=True,
                          nrow=3)

        #running_corrects += float(torch.sum(preds == labels.data))
    #acc = running_corrects / dataset_sizes
    #return acc
######################################################################
# Load Collected data Trained model
print('-------test-----------')
model, _, epoch = load_network(opt.name, opt)
print(model)
# Change to test mode
#model = model.eval()
if use_gpu:
    model = model.cuda()
    model = torch.nn.DataParallel(model)

# Extract feature
with torch.no_grad():
    acc = extract_feature(model, dataloaders['test'])
    print(acc)
