# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
import torch.nn.functional as F
matplotlib.use('agg')
import matplotlib.pyplot as plt
# from PIL import Image
import time
import os
import itertools
from losses import AngleLoss, ArcLoss
from model import ft_net, ft_net_ours, ft_net_dense, ft_net_EF4, ft_net_EF5, ft_net_EF6, ft_net_IR, ft_net_NAS, \
    ft_net_SE, ft_net_DSE, PCB, CPB, ft_net_angle, ft_net_arc, Autoencoder, Prediction
from random_erasing import RandomErasing
import yaml
from AugFolder import AugFolder,MyFolder
from shutil import copyfile
from loss_autocoder import crossview_contrastive_Loss
import random
from autoaugment import ImageNetPolicy
from utils import get_model_list, load_network, save_network, make_weights_for_balanced_classes
import pdb
from samplers import RandomIdentitySampler

version = torch.__version__
import numpy as np
from circle_loss import CircleLoss, convert_label_to_similarity

np.set_printoptions(threshold=np.inf)

# fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError:  # will be 3.x series
    print(
        'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
os.environ['CUDA_VISIBLE_DEVICES'] = '3,0'
# make the output
if not os.path.isdir('../mqveri/outputs'):
    os.mkdir('../mqveri/outputs')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--adam', action='store_true', help='use all training data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--init_name', default='imagenet', type=str, help='initial with ImageNet')
parser.add_argument('--data_dir', default='../mqveri_gan', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--train_veri', action='store_true', help='use part training data + veri')
parser.add_argument('--train_virtual', action='store_true', help='use part training data + virtual')
parser.add_argument('--train_comp', action='store_true', help='use part training data + comp')
parser.add_argument('--train_pku', action='store_true', help='use part training data + pku')
parser.add_argument('--train_comp_veri', action='store_true', help='use part training data + comp +veri')
parser.add_argument('--train_milktea', action='store_true', help='use part training data + com + veri+pku')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=36, type=int, help='batchsize')
parser.add_argument('--inputsize', default=256, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--pool', default='avg', type=str, help='last pool')
parser.add_argument('--autoaug', action='store_true', help='use Color Data Augmentation')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_NAS', action='store_true', help='use nasnetalarge')
parser.add_argument('--use_SE', action='store_true', help='use se_resnext101_32x4d')
parser.add_argument('--use_DSE', action='store_true', help='use senet154')
parser.add_argument('--use_IR', action='store_true', help='use InceptionResNetv2')
parser.add_argument('--use_EF4', action='store_true', help='use EF4')
parser.add_argument('--use_EF5', action='store_true', help='use EF5')
parser.add_argument('--use_EF6', action='store_true', help='use EF6')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50')
parser.add_argument('--CPB', action='store_true', help='use Center+ResNet50')
parser.add_argument('--fp16', action='store_true',
                    help='use float16 instead of float32, which will save about 50% memory')
parser.add_argument('--balance', action='store_true', help='balance sample')
parser.add_argument('--angle', action='store_true', help='use angle loss')
parser.add_argument('--arc', action='store_true', help='use arc loss')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--resume', action='store_true', help='use arc loss')
parser.add_argument('--circle', action='store_true', help='use Circle loss')
parser.add_argument('--gan', action='store_true', help='use Circle loss')
opt = parser.parse_args()
data_dir = opt.data_dir
name = opt.name
if not opt.resume:
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)
    opt.gpu_ids = gpu_ids

# set gpu ids
if len(opt.gpu_ids) > 0:
    cudnn.enabled = True
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#
SetRange = transforms.Lambda(lambda X: 2 * X - 1.)

transform_train_list = [
    transforms.Resize((round(opt.inputsize * 1.1), round(opt.inputsize * 1.1)), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]


if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list

transform_train_list_aug = [ImageNetPolicy()] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'train_aug': transforms.Compose(transform_train_list_aug),
}

image_datasets = {}

image_datasets['train'] = MyFolder(os.path.join(data_dir, 'train'),
                                               data_transforms['train'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              sampler=RandomIdentitySampler(image_datasets['train'], opt.batchsize,3),
                                              shuffle=False, num_workers=8, pin_memory=True) for x in ['train']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
num_train_imgs = len(image_datasets['train'])

use_gpu = torch.cuda.is_available()

y_loss = {}  # loss history
y_loss['train'] = []
y_err = {}
y_err['train'] = []


def train_model(autoencoder1, autoencoder2, img2txt, txt2img, model_reid, crossview_contrastive_Loss, optimizer,
                    start_epoch=0, num_epochs=500):
    since = time.time()
    gamma = 0.0  # auto_aug
    best_loss = 9999
    best_epoch = 0

    for epoch in range(num_epochs - start_epoch):
        epoch = epoch + start_epoch
        print('gamma: %.4f' % gamma)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                #scheduler.step()
                autoencoder1.train(True)
                autoencoder2.train(True)
                img2txt.train(True)
                txt2img.train(True) # Set model to training mode
            else:
                autoencoder1.train(False)
                autoencoder2.train(False)
                img2txt.train(False)
                txt2img.train(False) # Set model to evaluate mode

            loss_all, loss_rec1, loss_rec2, loss_cl, loss_pre, loss_gan = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels, label, camera_id = data
                view_ce = torch.where(camera_id == 2)[0]
                view_qian = torch.where(camera_id == 0)[0]
                view_hou = torch.where(camera_id == 1)[0]
                input_ce = torch.index_select(inputs, 0, view_ce)
                input_qian = torch.index_select(inputs, 0, view_qian)
                input_hou = torch.index_select(inputs, 0, view_hou)
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue

                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    input_ce = Variable(input_ce.cuda().detach())
                    input_qian = Variable(input_qian.cuda().detach())
                    input_hou = Variable(input_hou.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                    input_ce = Variable(input_ce)
                    input_qian = Variable(input_qian)
                    input_hou = Variable(input_hou)

                # forward
                qian_f = model_reid(input_qian)
                ce_f = model_reid(input_ce)
                z_1 = autoencoder1.module.encoder(qian_f)
                z_2 = autoencoder2.module.encoder(ce_f)

                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(autoencoder1.module.decoder(z_1), qian_f)
                recon2 = F.mse_loss(autoencoder2.module.decoder(z_2), ce_f)
                reconstruction_loss = recon1 + recon2

                # Cross-view Contrastive_Loss
                cl_loss = crossview_contrastive_Loss(z_1, z_2, 9)

                # Cross-view Dual-Prediction Loss
                qian2ce, _ = img2txt(z_1)
                ce2qian, _ = txt2img(z_2)
                pre1 = F.mse_loss(qian2ce, z_2)
                pre2 = F.mse_loss(ce2qian, z_1)
                dualprediction_loss = (pre1 + pre2)
                #保证预测出来特征与解码之后的特征相近
                gan1 = F.mse_loss(ce_f, autoencoder2.module.decoder(qian2ce))
                gan2 = F.mse_loss(qian_f, autoencoder1.module.decoder(ce2qian))
                dualgan_loss = (gan1 + gan2)

                loss = cl_loss + reconstruction_loss * 0.1

                # we train the autoencoder by L_cl and L_rec first to stabilize
                # the training of the dual prediction
                if epoch >= 100:
                    loss = loss + dualprediction_loss * 0.1 + dualgan_loss * 0.1
                # backward + optimize only if in training phase
                # zero the parameter gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

                # print('Iteration: loss:%.2f accuracy:%.2f'%(loss.item(), float(torch.sum(preds == labels.data))/now_batch_size ) )
                # statistics

                loss_all += loss.item() * now_batch_size / 3
                loss_rec1 += recon1.item() * now_batch_size / 3
                loss_rec2 += recon2.item() * now_batch_size / 3
                loss_pre += dualprediction_loss.item() * now_batch_size / 3
                loss_gan += dualgan_loss.item() * now_batch_size / 3
                loss_cl += cl_loss.item() * now_batch_size / 3
                del (loss, qian_f, ce_f)

            epoch_loss = loss_all / dataset_sizes[phase]
            epoch_loss_rec1 = loss_rec1 / dataset_sizes[phase]
            epoch_loss_rec2 = loss_rec2 / dataset_sizes[phase]
            epoch_loss_pre = loss_pre / dataset_sizes[phase]
            epoch_loss_cl = loss_cl / dataset_sizes[phase]
            epoch_loss_gan = loss_gan / dataset_sizes[phase]

            print('{} Loss: {:.4f} rec1_Loss: {:.4f} rec2: {:.4f} pre: {:.4f} cl: {:.4f} gan: {:.4f}'.format(
                phase, epoch_loss, epoch_loss_rec1, epoch_loss_rec2, epoch_loss_pre, epoch_loss_cl, epoch_loss_gan))

            y_loss[phase].append(epoch_loss)
            # deep copy the model
            if len(opt.gpu_ids) > 1 and epoch % 24 == 0:
                save_network(autoencoder1.module, "encoder_qian_ce_loss_1", epoch + 1)
                save_network(autoencoder2.module, "encoder_ce_qian_loss_1", epoch + 1)
                save_network(img2txt.module, "qian2ce_loss_1", epoch + 1)
                save_network(txt2img.module, "ce2qian_loss_1", epoch + 1)
            elif epoch % 24 == 0:
                save_network(autoencoder1, "encoder_qian_ce_loss_1", epoch + 1)
                save_network(autoencoder2, "encoder_ce_qian_loss_1", epoch + 1)
                save_network(img2txt, "qian2ce_loss_1", epoch + 1)
                save_network(txt2img, "ce2qian_loss_1", epoch + 1)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            last_model_wts_encoder1 = autoencoder1.state_dict()
            last_model_wts_encoder2 = autoencoder2.state_dict()
            last_model_wts_img2txt = img2txt.state_dict()
            last_model_wts_txt2img = txt2img.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:d} Best Train Loss: {:4f}'.format(best_epoch, best_loss))

    # load best model weights
    autoencoder1.load_state_dict(last_model_wts_encoder1)
    autoencoder2.load_state_dict(last_model_wts_encoder2)
    img2txt.load_state_dict(last_model_wts_img2txt)
    txt2img.load_state_dict(last_model_wts_txt2img)

    save_network(autoencoder1, "encoder_qian_ce_loss_1", 'last')
    save_network(autoencoder2, "encoder_ce_qian_loss_1", 'last')
    save_network(img2txt, "qian2ce_loss_1", 'last')
    save_network(txt2img, "ce2qian_loss_1", 'last')
    return autoencoder1



######################################################################
# Draw Curve
# ---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    # ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    # ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('../mqveri/outputs', name, 'train.png'))


######################################################################
autoencoder1 = Autoencoder(encoder_dim=[512, 1024, 1024, 1024, 128], activation='relu', batchnorm=True)
autoencoder2 = Autoencoder(encoder_dim=[512, 1024, 1024, 1024, 128], activation='relu', batchnorm=True)

img2txt = Prediction(prediction_dim=[128,128,256,128])   #表示两种不同的视角
txt2img = Prediction(prediction_dim=[128,128,256,128])
##########################
# Put model parameter in front of the optimizer!!!
print(autoencoder1)
print(img2txt)

if len(opt.gpu_ids) > 1:
    autoencoder1 = torch.nn.DataParallel(autoencoder1, device_ids=opt.gpu_ids).cuda()
    autoencoder2 = torch.nn.DataParallel(autoencoder2, device_ids=opt.gpu_ids).cuda()
    img2txt = torch.nn.DataParallel(img2txt, device_ids=opt.gpu_ids).cuda()
    txt2img = torch.nn.DataParallel(txt2img, device_ids=opt.gpu_ids).cuda()
else:
    autoencoder1 = autoencoder1.cuda()
    autoencoder2 = autoencoder2.cuda()
    img2txt = img2txt.cuda()
    txt2img = txt2img.cuda()

optimizer_ft = optim.Adam(itertools.chain(autoencoder1.parameters(), autoencoder2.parameters(),
                            img2txt.parameters(), txt2img.parameters()), lr=0.0001)

# Decay LR by a factor of 0.1 every 40 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
#exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[60, 75], gamma=0.95)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
#
dir_name1 = os.path.join('../mqveri/outputs', "encoder_qian_ce_loss_1")
dir_name2 = os.path.join('../mqveri/outputs', "encoder_ce_qian_loss_1")
dir_name3 = os.path.join('../mqveri/outputs', "qian2ce_loss_1")
dir_name4 = os.path.join('../mqveri/outputs', "ce2qian_loss_1")

if not opt.resume:
    if not os.path.isdir(dir_name1):
        os.mkdir(dir_name1)
    if not os.path.isdir(dir_name2):
        os.mkdir(dir_name2)
    if not os.path.isdir(dir_name3):
        os.mkdir(dir_name3)
    if not os.path.isdir(dir_name4):
        os.mkdir(dir_name4)
    # record every run
    copyfile('./train_autocoder.py', dir_name1 + '/train.py')
    copyfile('./model.py', dir_name1 + '/model.py')
    # save opts
    with open('%s/opts.yaml' % dir_name1, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

#加载训练好的Reid模型
model_reid = ft_net_ours(54, 0.5, 2, None, 'avg')
reid_path = os.path.join('../mqveri/outputs', "mmtm_8", "net_last.pth")
model_reid.load_state_dict(torch.load(reid_path))
model_reid.classifier.classifier = nn.Sequential()
model_reid = model_reid.cuda()
model_reid = torch.nn.DataParallel(model_reid)
model_reid = model_reid.module
model_reid = model_reid.eval()
#####################################################
model = train_model(autoencoder1, autoencoder2, img2txt, txt2img, model_reid, crossview_contrastive_Loss, optimizer_ft,
                    start_epoch=0, num_epochs=500)

