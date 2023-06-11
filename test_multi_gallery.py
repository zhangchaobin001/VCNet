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

from AugFolder import MyFolder
from samplers import RandomIdentitySampler,RandomIdentity
from utils import load_network
from tqdm import tqdm
from model import ft_net, ft_net_angle, ft_net_EF4, ft_net_EF5, ft_net_EF6, ft_net_arc, ft_net_IR, ft_net_SE, \
    ft_net_DSE, ft_net_dense, ft_net_NAS, PCB, PCB_test, CPB, qian_aware,ce_aware,hou_aware
from evaluate_gpu import calculate_result
from evaluate_rerank import calculate_result_rerank
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
parser.add_argument('--name', default='ft_circle', type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
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
parser.add_argument('--ours', action='store_true', help='use ft_net_ours.' )
parser.add_argument('--ibn', action='store_true', help='use resnet+ibn' )
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
#opt.ours = config['ours']

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

opt.nclasses = config['nclasses']
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
        transforms.Resize((round(opt.inputsize), round(opt.inputsize)), interpolation=3),
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

if opt.multi:
    image_datasets = {x: MyFolder(os.path.join(data_dir, x), data_transforms) for x in
                      ['gallery', 'query', 'multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=20) for x in
                   ['gallery', 'query', 'multi-query']}
else:
    image_datasets = {x: MyFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query']}
    dataloaders1 = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=20, drop_last=True) for x in ['gallery']}
    #dataloaders2 = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, sampler=RandomIdentity(image_datasets[x],3),shuffle=False, num_workers=20) for x in ['query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
# ---------------------------
'''
def load_network(network):
    which_epoch = opt.which_epoch
    if which_epoch.isdigit():
        save_path = os.path.join('../mqveri/outputs',name,'net_%03d.pth'%int(which_epoch))
    else:
        save_path = os.path.join('../mqveri/outputs',name,'net_%s.pth'%which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network
'''


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


def extract_feature(model, model_view, model_backbone, model_qian, model_hou, model_ce, dataloaders):
    features = torch.FloatTensor()
    features_view = torch.FloatTensor()
    count = 0
    cameras = []
    labels = []
    for data in tqdm(dataloaders):
        img, target, label, camera_id = data
        label = label.numpy().tolist()
        camera_id = camera_id.numpy().tolist()
        labels = labels + label
        cameras = cameras + camera_id
        n, c, h, w = img.size()
        count += n
        # print(count)
        ff = torch.FloatTensor(n, 512).zero_().cuda()
        ff_view = torch.FloatTensor(n, 512).zero_().cuda()
        input_img = Variable(img.cuda())

        outputs_view = model_view(input_img)
        # outputs_view = f
        # f = model_backbone(input_img)
        # _, f_qian = model_qian(f)
        # _, f_hou = model_hou(f)
        # _, f_ce = model_ce(f)
        # _, outputs = model(f, f_qian, f_hou, f_ce)
        #outputs = model(input_img)
        outputs = model(input_img, outputs_view)

        ff += outputs
        ff_view += outputs_view
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        fnorm_view = torch.norm(ff_view, p=2, dim=1, keepdim=True)
        ff_view = ff_view.div(fnorm_view.expand_as(ff_view))
        # print(ff.shape)
        features = torch.cat((features, ff.data.cpu().float()), 0)
        features_view = torch.cat((features_view, ff_view.data.cpu().float()), 0)
    labels = np.asarray(labels)
    cameras = np.asarray(cameras)
    return features, features_view, labels, cameras


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        # filename = path.split('/')[3]
        filename = os.path.basename(path)
        label = filename.split('_')[0]
        camera = filename.split('_')[-1].split(".")[0]
        labels.append(int(label))
        camera_id.append(int(camera))
    return camera_id, labels

'''
def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        # Test Gallery Image
        if not 'c' in filename: 
            labels.append(9999999)
            camera_id.append(9999999)
        else:
            label = filename[0:6]
            camera = filename.split('c')[1]
            if label[0:2]=='-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera[0:3]))
        #print(camera[0:3])
    return camera_id, labels
'''

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

#gallery_cam, gallery_label = get_id(gallery_path)
#query_cam, query_label = get_id(query_path)

if opt.multi:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_cam, mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
model, _, epoch = load_network(opt.name, opt)
model_view = ft_net(3, droprate=0, stride=1, circle = False, ibn=False)
view_path = os.path.join('../mqveri/outputs', "viewagain", "net_last.pth")
# Change to test mode
#model_view = torch.nn.DataParallel(model_view)
model_view.load_state_dict(torch.load(view_path))
model_view.classifier.classifier = nn.Sequential()
model_view = model_view.cuda()
#model_view = model_view.module
model_view = model_view.eval()
#model_view, _, epoch = load_network("resnet_view", opt_view)
# Remove the final fc layer and classifier layer

model_backbone = ft_net(124, 0, 2, circle =False, ibn=False)
backbone_path = os.path.join('../mqveri/outputs', "ft124", "net_last.pth")
model_backbone.load_state_dict(torch.load(backbone_path))
model_backbone.classifier.classifier = nn.Sequential()
model_backbone = model_backbone.cuda()
model_backbone = model_backbone.eval()

model_qian = qian_aware(124)
qian_path = os.path.join('../mqveri/outputs', "qian_aware", "net_last.pth")
model_qian.load_state_dict(torch.load(qian_path))
model_qian.classifier.classifier = nn.Sequential()
model_qian = model_qian.cuda()
model_qian = model_qian.eval()

model_hou = hou_aware(124)
hou_path = os.path.join('../mqveri/outputs', "hou_aware", "net_last.pth")
model_hou.load_state_dict(torch.load(hou_path))
model_hou.classifier.classifier = nn.Sequential()
model_hou = model_hou.cuda()
model_hou = model_hou.eval()

model_ce = ce_aware(124)
ce_path = os.path.join('../mqveri/outputs', "ce_aware", "net_last.pth")
model_ce.load_state_dict(torch.load(ce_path))
model_ce.classifier.classifier = nn.Sequential()
model_ce = model_ce.cuda()
model_ce = model_ce.eval()

#用不上暂时去掉
if opt.PCB:
    model = PCB_test(model)
elif opt.CPB:
    # model.classifier0 = nn.Sequential()
    # model.classifier1 = nn.Sequential()
    # model.classifier2 = nn.Sequential()
    # model.classifier3 = nn.Sequential()
    model.classifier0.classifier = nn.Sequential()
    model.classifier1.classifier = nn.Sequential()
    model.classifier2.classifier = nn.Sequential()
    model.classifier3.classifier = nn.Sequential()
    # model[1].model.fc = nn.Sequential()
    # model[1].classifier.classifier = nn.Sequential()
else:
    model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
#model_view = model_view.eval()
if use_gpu:
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    #model_view = model_view.cuda()
    #model_view = torch.nn.DataParallel(model_view)
# Extract feature
with torch.no_grad():
    #query_feature, query_view,query_label, query_cam = extract_feature(model, model_view, dataloaders2['query'])
    gallery_feature, gallery_view,gallery_label, gallery_cam= extract_feature(model, model_view, model_backbone, model_qian, model_hou, model_ce, dataloaders1['gallery'])

print('Gallery Size: %d' % len(gallery_label))
#print('Query Size: %d' % len(query_label))

# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_v':gallery_view.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam}
scipy.io.savemat('pytorch_result_gallery_ft.mat',result)

result_file = '../mqveri/outputs/%s/result.txt' % opt.name
