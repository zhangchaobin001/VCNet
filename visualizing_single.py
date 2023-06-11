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
import re
import copy
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=6, type=int, help='test_image_index')
parser.add_argument('--test_dir',default='../mqveri',type=str, help='./test_data')
opts = parser.parse_args()

data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    #pdb.set_trace()
    im = cv.resize(im, (256, 256))
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
#result_gallery = scipy.io.loadmat('pytorch_result_gallery.mat')
#result = scipy.io.loadmat('pytorch_result.mat')
result_gallery = scipy.io.loadmat('pytorch_result_gallery_cal_160.mat')
result = scipy.io.loadmat('pytorch_result_cal_160.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
query_view = torch.FloatTensor(result['query_v'])
gallery_feature = torch.FloatTensor(result_gallery['gallery_f'])
gallery_cam = result_gallery['gallery_cam'][0]
gallery_view = torch.FloatTensor(result_gallery['gallery_v'])
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
def sort_img(qf, ql, qc, gf, gl, gc, qv, gv):
    query = qf.view(-1, 1)  # torch.Size([512, 1])
    # print(query.shape)
    score = torch.mm(gf, query)  # torch.Size([13500, 1])
    score = score.squeeze(1).cpu()  # torch.Size([13500])
    score = score.numpy()  # numpy array
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    #same camera
    camera_index = np.argwhere(gc==qc)
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
            simx_y = torch.mm(gv[cam[k]].view(1, -1), gv[cam[i]].view(-1, 1))
            #import pdb
            #pdb.set_trace()
            if (x == y) and (simx_y > 0.6):
                mask[np.argwhere(index == cam[k])] = False
    
    index = index[mask]

    return index1

i = opts.query_index
#import pdb
#pdb.set_trace()
print(query_label)

index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam, query_view, gallery_view)

########################################################################
# Visualize the rank result

query_path, _ = image_datasets['query'].imgs[i]
querylabel = query_label[i]
import pdb
#pdb.set_trace()
querycam = query_cam[i]
match = str(querylabel) + '[0-9_]*' + str(querycam) + '.jpg$'

filedir = '../mqveri/query/' + str(querylabel)
filelist = os.listdir(filedir)
filename = []
for i in filelist:
    if re.match(match, i):
        query_path = os.path.join(filedir, i)
print(query_path)
print('Top 10 images are as follow:')
try: # Visualize Ranking Result
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(10,4))
    ax = plt.subplot(3,7,1)
    ax.axis('off')
    imshow(query_path,'query')
    for i in range(20):
        ax = plt.subplot(3,7,i+2)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0.25)
        ax.axis('off')
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        #label = gallery_label[index[i]]
        label = int(img_path.split('/')[3])
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

fig.savefig("show_csm%d.png" % opts.query_index)