import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2 as cv
import pdb
import copy
import re
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=0, type=int, help='test_image_index')
parser.add_argument('--test_dir',default='../mqveri',type=str, help='./test_data')
opts = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    im = cv.resize(im, (256, 256))
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
result_gallery = scipy.io.loadmat('pytorch_result_gallery_cal_160.mat')
result = scipy.io.loadmat('pytorch_result_cal_160.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result_gallery['gallery_f'])
gallery_cam = result_gallery['gallery_cam'][0]
gallery_label = result_gallery['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

#######################################################################
# sort the images
def sort_img(qf1,qf2,qf3, ql, qc1, qc2, qc3, gf, gl, gc):
    query1 = qf1.view(-1, 1)  # torch.Size([512, 1])
    query2 = qf2.view(-1, 1)
    query3 = qf3.view(-1, 1)
    # print(query.shape)
    score1 = torch.mm(gf, query1)  # torch.Size([13500, 1])
    score1 = score1.squeeze(1).cpu()  # torch.Size([13500])
    score1 = score1.numpy()  # numpy array
    score2 = torch.mm(gf, query2)  # torch.Size([13500, 1])
    score2 = score2.squeeze(1).cpu()  # torch.Size([13500])
    score2 = score2.numpy()  # numpy array
    score3 = torch.mm(gf, query3)  # torch.Size([13500, 1])
    score3 = score3.squeeze(1).cpu()  # torch.Size([13500])
    score3 = score3.numpy()  # numpy array
    # predict index
    score = (score1 + score2 + score3) / 3
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    #same camera
    camera_index = np.argwhere((gc==qc1) | (gc == qc2) | (gc == qc3))

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    mask = np.in1d(index, junk_index, invert=True)
    index1 = index[mask]

    mask_good = np.in1d(index1, good_index)
    mask_cam = copy.deepcopy(mask_good)
    cam = index1[mask_cam]
    for i in range(len(cam) - 1):
        x = gc[cam[i]]
        for k in range(i + 1, len(cam)):
            y = gc[cam[k]]
            simx_y = torch.mm(gf[cam[k]].view(1, -1), gf[cam[i]].view(-1, 1))
            # import pdb
            # pdb.set_trace()
            if (x == y) and (simx_y > 0.6):
                mask[np.argwhere(index == cam[k])] = False

    index = index[mask]

    return index1

i = opts.query_index
print(query_label)

if i%3==0:
    index = sort_img(query_feature[i],query_feature[i+1],query_feature[i+2], query_label[i], query_cam[i], query_cam[i+1], query_cam[i+2], gallery_feature, gallery_label,
                           gallery_cam)
elif i%3==1:
    index = sort_img(query_feature[i-1], query_feature[i + 1], query_feature[i], query_label[i],
                               query_cam[i], query_cam[i-1], query_cam[i+1], gallery_feature, gallery_label,
                               gallery_cam)
else:
    index = sort_img(query_feature[i - 1], query_feature[i-2], query_feature[i], query_label[i],
                               query_cam[i], query_cam[i-1], query_cam[i-2], gallery_feature, gallery_label,
                               gallery_cam)
########################################################################
# Visualize the rank result

#pdb.set_trace()

if i%3 == 0:
    #query_path1, _ = image_datasets['query'].imgs[i]
    #query_path2, _ = image_datasets['query'].imgs[i+1]
    #query_path3, _ = image_datasets['query'].imgs[i+2]
    querylabel = query_label[i]
    querycam1 = query_cam[i]
    querycam2 = query_cam[i+1]
    querycam3 = query_cam[i+2]
elif i%3 == 1:
    #query_path1, _ = image_datasets['query'].imgs[i]
    #query_path2, _ = image_datasets['query'].imgs[i+1]
    #query_path3, _ = image_datasets['query'].imgs[i-1]
    querylabel = query_label[i]
    querycam1 = query_cam[i]
    querycam2 = query_cam[i + 1]
    querycam3 = query_cam[i - 1]
else:
    #query_path1, _ = image_datasets['query'].imgs[i]
    #query_path2, _ = image_datasets['query'].imgs[i-1]
    #query_path3, _ = image_datasets['query'].imgs[i-2]
    querylabel = query_label[i]
    querycam1 = query_cam[i]
    querycam2 = query_cam[i - 2]
    querycam3 = query_cam[i - 1]
match1 = str(querylabel) + '[0-9_]*' + str(querycam1) + '.jpg$'
match2 = str(querylabel) + '[0-9_]*' + str(querycam2) + '.jpg$'
match3 = str(querylabel) + '[0-9_]*' + str(querycam3) + '.jpg$'
filedir = '../mqveri/query/' + str(querylabel)
filelist = os.listdir(filedir)
filename = []
'''
for i in filelist:
    if re.match(match1, i):
        query_path1 = os.path.join(filedir, i)
    if re.match(match2, i):
        query_path2 = os.path.join(filedir, i)
    if re.match(match3, i):
        query_path3 = os.path.join(filedir, i)
'''
query_path1 = os.path.join(filedir, filelist[0])
query_path2 = os.path.join(filedir, filelist[1])
query_path3 = os.path.join(filedir, filelist[2])
#pdb.set_trace()
#query_path1, _ = image_datasets['query'].imgs[i]
#query_path2, _ = image_datasets['query'].imgs[i+1]
print(query_path1)
print(query_path2)
print(query_path3)
print('Top 10 images are as follow:')
try: # Visualize Ranking Result
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(10,4))
    ax = plt.subplot(3,8,1)
    ax.axis('off')
    imshow(query_path1,'query1')
    ax = plt.subplot(3, 8, 2)
    ax.axis('off')
    imshow(query_path2,'query2')
    ax = plt.subplot(3, 8, 3)
    ax.axis('off')
    imshow(query_path3,'query3')
    for i in range(20):
        ax = plt.subplot(3,8,i+4)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0.25)
        ax.axis('off')
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        #label = gallery_label[index[i]]
        label = int(img_path.split('/')[3])
        print(label)
        imshow(img_path)
        if label == querylabel:
            ax.set_title('%d'%(i+1), color='green')
        else:
            ax.set_title('%d'%(i+1), color='red')
        print(img_path)
except RuntimeError:
    for i in range(10):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

fig.savefig("show_no%d.png" % opts.query_index)