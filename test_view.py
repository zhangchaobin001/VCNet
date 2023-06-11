# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
from model import ft_net
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
from utilsView import load_network
from AugFolder import AugFolder, ViewFolder, ViewAugFolder
from tqdm import tqdm
import pdb


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
parser.add_argument('--ibn', action='store_true', help='use resnet+ibn' )
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
opt.ibn = config['ibn']
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
if opt.h == opt.w:
    data_transforms = transforms.Compose([
        transforms.Resize((opt.inputsize, opt.inputsize), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
else:
    data_transforms = transforms.Compose([
        transforms.Resize((round(opt.h * 1.1), round(opt.w * 1.1)), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print(opt.h)

data_dir = test_dir
image_datasets = {}
image_datasets['test'] = ViewFolder(os.path.join(data_dir, 'query_miss_ce'),
                                        data_transforms)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=1, pin_memory=True)
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

'''
def extract_feature(model, dataloaders):
    running_corrects = 0.0
    for data in tqdm(dataloaders):  # 读取信息的进度条
        inputs, labels = data  # 一个batch的data
        labels = labels.long()
        if use_gpu:
            inputs = Variable(inputs.cuda().detach())
            labels = Variable(labels.cuda().detach())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        running_corrects += float(torch.sum(preds == labels.data))
    acc = running_corrects / dataset_sizes
    return acc
'''
def extract_feature(model, dataloaders):
    running_corrects = 0.0
    dir = '/data/cbzhang4/mqveri/query_miss_ce'
    for data in tqdm(dataloaders):  # 读取信息的进度条
        inputs, filename, view= data  # 一个batch的data
        #labels = labels.long()
        inputs = Variable(inputs.cuda().detach())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        pdb.set_trace()

        #running_corrects += float(torch.sum(preds == labels.data))
    #acc = running_corrects / dataset_sizes
    #return acc
######################################################################
# Load Collected data Trained model
print('-------test-----------')
#model, _, epoch = load_network(opt.name, opt)
model = ft_net(3, droprate=0, stride = 1, ibn = opt.ibn)
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
