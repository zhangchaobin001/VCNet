# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import math
from collections import Counter, defaultdict
from loss_autocoder import crossview_contrastive_Loss
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import pdb
import yaml
np.set_printoptions(threshold=np.inf)
from AugFolder import MyFolder
from samplers import RandomIdentitySamplerGan,RandomIdentity, RandomIdentityGan,RandomIdentitySampler
from utils import load_network
from tqdm import tqdm
from model import ft_net, ft_net_angle, PCB, PCB_test, CPB, VanillaVAE, Autoencoder, Prediction, qian_aware,ce_aware,hou_aware, three_select
from evaluate_gpu import calculate_result
from evaluate_rerank import calculate_result_rerank
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
parser.add_argument('--batchsize', default=6, type=int, help='batchsize')  #需要固定为3
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
opt_view = parser.parse_args()
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


opt.nclasses = config['nclasses']
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
print(data_dir)

if opt.multi:
    image_datasets = {x: MyFolder(os.path.join(data_dir, x), data_transforms) for x in
                      ['gallery', 'query', 'multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=20) for x in
                   ['gallery', 'query', 'multi-query']}
else:
    image_datasets = {x: MyFolder(os.path.join(data_dir, x), data_transforms) for x in ['query_multi']}
    dataloaders2 = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, sampler=RandomIdentityGan(image_datasets[x], num_instances=6),shuffle=False, num_workers=20) for x in ['query_multi']}
class_names = image_datasets['query_multi'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
# ---------------------------

######################################################################
#加载生成网络，前到后
model_gan_qian_hou_encoder1 = Autoencoder(encoder_dim=[512, 1024, 1024, 1024, 128], activation='relu', batchnorm=True)
model_gan_hou_qian_encoder2 = Autoencoder(encoder_dim=[512, 1024, 1024, 1024, 128], activation='relu', batchnorm=True)
save_path_qian_hou_encoder1 = os.path.join('../mqveri/outputs', "encoder_qian_hou_loss_1", "net_last.pth")
save_path_hou_qian_encoder2 = os.path.join('../mqveri/outputs', "encoder_hou_qian_loss_1", "net_last.pth")

model_qian2hou = Prediction(prediction_dim=[128,128,256,128])
model_hou2qian = Prediction(prediction_dim=[128,128,256,128])
save_path_qian2hou = os.path.join('../mqveri/outputs', "qian2hou_loss_1", "net_last.pth")
save_path_hou2qian = os.path.join('../mqveri/outputs', "hou2qian_loss_1", "net_last.pth")

model_gan_qian_hou_encoder1 = torch.nn.DataParallel(model_gan_qian_hou_encoder1)
model_gan_hou_qian_encoder2 = torch.nn.DataParallel(model_gan_hou_qian_encoder2)
model_qian2hou = torch.nn.DataParallel(model_qian2hou)
model_hou2qian = torch.nn.DataParallel(model_hou2qian)

model_gan_qian_hou_encoder1.load_state_dict(torch.load(save_path_qian_hou_encoder1))
model_gan_hou_qian_encoder2.load_state_dict(torch.load(save_path_hou_qian_encoder2))
model_qian2hou.load_state_dict(torch.load(save_path_qian2hou))
model_hou2qian.load_state_dict(torch.load(save_path_hou2qian))

model_gan_qian_hou_encoder1 = model_gan_qian_hou_encoder1.cuda()
model_gan_hou_qian_encoder2 = model_gan_hou_qian_encoder2.cuda()
model_qian2hou = model_qian2hou.cuda()
model_hou2qian = model_hou2qian.cuda()

model_gan_qian_hou_encoder1 = model_gan_qian_hou_encoder1.module
model_gan_hou_qian_encoder2 = model_gan_hou_qian_encoder2.module
model_qian2hou = model_qian2hou.module
model_hou2qian = model_hou2qian.module

model_gan_qian_hou_encoder1 = model_gan_qian_hou_encoder1.eval()
model_gan_hou_qian_encoder2 = model_gan_hou_qian_encoder2.eval()
model_qian2hou = model_qian2hou.eval()
model_hou2qian = model_hou2qian.eval()

#加载生成网络，前到侧
########################################################################

model_gan_qian_ce_encoder1 = Autoencoder(encoder_dim=[512, 1024, 1024, 1024, 128], activation='relu', batchnorm=True)
model_gan_ce_qian_encoder2 = Autoencoder(encoder_dim=[512, 1024, 1024, 1024, 128], activation='relu', batchnorm=True)
save_path_qian_ce_encoder1 = os.path.join('../mqveri/outputs', "encoder_qian_ce_loss_1", "net_last.pth")
save_path_ce_qian_encoder2 = os.path.join('../mqveri/outputs', "encoder_ce_qian_loss_1", "net_last.pth")

model_qian2ce = Prediction(prediction_dim=[128,128,256,128])
model_ce2qian = Prediction(prediction_dim=[128,128,256,128])
save_path_qian2ce = os.path.join('../mqveri/outputs', "qian2ce_loss_1", "net_last.pth")
save_path_ce2qian = os.path.join('../mqveri/outputs', "ce2qian_loss_1", "net_last.pth")

model_gan_qian_ce_encoder1 = torch.nn.DataParallel(model_gan_qian_ce_encoder1)
model_gan_ce_qian_encoder2 = torch.nn.DataParallel(model_gan_ce_qian_encoder2)
model_qian2ce = torch.nn.DataParallel(model_qian2ce)
model_ce2qian = torch.nn.DataParallel(model_ce2qian)

model_gan_qian_ce_encoder1.load_state_dict(torch.load(save_path_qian_ce_encoder1))
model_gan_ce_qian_encoder2.load_state_dict(torch.load(save_path_ce_qian_encoder2))
model_qian2ce.load_state_dict(torch.load(save_path_qian2ce))
model_ce2qian.load_state_dict(torch.load(save_path_ce2qian))

model_gan_qian_ce_encoder1 = model_gan_qian_ce_encoder1.cuda()
model_gan_ce_qian_encoder2 = model_gan_ce_qian_encoder2.cuda()
model_qian2ce = model_qian2ce.cuda()
model_ce2qian = model_ce2qian.cuda()

model_gan_qian_ce_encoder1 = model_gan_qian_ce_encoder1.module
model_gan_ce_qian_encoder2 = model_gan_ce_qian_encoder2.module
model_qian2ce = model_qian2ce.module
model_ce2qian = model_ce2qian.module

model_gan_qian_ce_encoder1 = model_gan_qian_ce_encoder1.eval()
model_gan_ce_qian_encoder2 = model_gan_ce_qian_encoder2.eval()
model_qian2ce = model_qian2ce.eval()
model_ce2qian = model_ce2qian.eval()

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

def gan_img(cameras, ff, img):
    count = dict(Counter(cameras))
    ff_gan = copy.deepcopy(ff)
    repe = [key for key, value in count.items() if value > 1]
    re_index = cameras.index(repe[0])
    view_miss = defaultdict(str)

    for i in range(3):
        if i != re_index:
            view, _ = model_view(torch.unsqueeze(img[i], dim=0))
            _, preds = torch.max(view.data, 1)
            view_miss[i] = preds
    #pdb.set_trace()
    if 1 not in view_miss.values():
        for item, value in view_miss.items():
            if value == 0:
                z1 = model_gan_qian_hou_encoder1.encoder(torch.unsqueeze(ff[item], dim=0))
                gan_hou, _ = model_qian2hou(z1)
                ganhou = model_gan_hou_qian_encoder2.decoder(gan_hou)
                ff_gan[re_index] = torch.squeeze(ganhou, dim=0)
                break
        #print("miss hou")
    '''
    if 2 not in view_miss.values() and 5 not in view_miss.values():
        for item, value in view_miss.items():
            if value == 0:
                z1 = model_gan_qian_ce_encoder1.encoder(torch.unsqueeze(ff[item], dim=0))
                gan_ce, _ = model_qian2ce(z1)
                gance = model_gan_ce_qian_encoder2.decoder(gan_ce)
                ff_gan[re_index] = torch.squeeze(gance, dim=0)
                break
        print("miss ce")
    if 0 not in view_miss.values() and 3 not in view_miss.values() and 6 not in view_miss.values():
        for item, value in view_miss.items():
            if value == 1:
                z1 = model_gan_hou_qian_encoder2.encoder(torch.unsqueeze(ff[item], dim=0))
                gan_qian, _ = model_hou2qian(z1)
                ganqian = model_gan_qian_hou_encoder1.decoder(gan_qian)
                ff_gan[re_index] = torch.squeeze(ganqian, dim=0)
                break
        print("miss qian")
    '''
    return cameras, ff_gan, re_index

def extract_feature(model, model_view, dataloaders):
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
        #pdb.set_trace()
        set_lst = set(camera_id)
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n, 512).zero_().cuda()
        ff_view = torch.FloatTensor(n, 512).zero_().cuda()

        input_img = Variable(img.cuda())

        '''
        if (len(set_lst) != len(camera_id)):
            camera_id, outputs_gan, re_index = gan_img(camera_id, outputs, input_img)
            camera_id[re_index] = 1000
            outputs = outputs_gan
        '''

        outputs_view = model_view(input_img)
        #outputs_view = x[1]
        #outputs = model(input_img)
        # f = model_backbone(input_img)
        # _, f_qian = model_qian(f)
        # _, f_hou = model_hou(f)
        # _, f_ce = model_ce(f)
        # _, outputs = model(f, f_qian, f_hou, f_ce)
        outputs = model(input_img, outputs_view)

        ff += outputs
        ff_view += outputs_view

        # norm feature
        cameras = cameras + camera_id
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

query_path = image_datasets['query_multi'].imgs

#gallery_cam, gallery_label = get_id(gallery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')

model_view = ft_net(3, droprate=0, stride=1, circle = False, ibn=False)
view_path = os.path.join('../mqveri/outputs', "viewagain", "net_last.pth")
# Change to test mode
#model_view = torch.nn.DataParallel(model_view)
model_view.load_state_dict(torch.load(view_path))
model_view.classifier.classifier = nn.Sequential()
model_view = model_view.cuda()
#model_view = model_view.module
model_view = model_view.eval()
# Remove the final fc layer and classifier layer
'''
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
'''
model, _, epoch = load_network(opt.name, opt)
#用不上的暂时去掉
if opt.PCB:
    model = PCB_test(model)
elif opt.CPB:
    model.classifier0.classifier = nn.Sequential()
    model.classifier1.classifier = nn.Sequential()
    model.classifier2.classifier = nn.Sequential()
    model.classifier3.classifier = nn.Sequential()
else:
    model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()
    model = torch.nn.DataParallel(model)

# Extract feature
with torch.no_grad():
    query_feature, query_view,query_label, query_cam = extract_feature(model, model_view, dataloaders2['query_multi'])
    #gallery_feature, gallery_view,gallery_label, gallery_cam= extract_feature(model, model_view, dataloaders1['gallery'])

print('Query Size: %d' % len(query_label))

result = {'query_f': query_feature.numpy(), 'query_v': query_view.numpy(),
          'query_label': query_label, 'query_cam': query_cam}

scipy.io.savemat('pytorch_result_query_ft.mat', result)

result_file = '../mqveri/outputs/%s/result.txt' % opt.name
