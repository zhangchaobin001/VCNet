# -*- coding: utf-8 -*-

import scipy.io
import torch
import numpy as np
# import time
import os
import pickle
import pdb
import sys
import copy
import pdb
np.set_printoptions(threshold=sys.maxsize)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#######################################################################
def softmax(x):
    x = x.astype(float)
    if x.ndim == 1:
        S = np.sum(x)
        return x / S
    elif x.ndim == 2:
        M, N = x.shape
        for n in range(N):
            S = np.sum(x[:, n])
            x[:, n] = x[:, n] / S
        return x
    else:
        print("The input array is not 1- or 2-dimensional.")
'''
def softmax(x):
    x = x.astype(float)
    if x.ndim == 1:
        S = np.sum(np.exp(x))
        return np.exp(x) / S
    elif x.ndim == 2:
        result = np.zeros_like(x)
        M, N = x.shape
        for n in range(N):
            S = np.sum(np.exp(x[:, n]))
            result[:, n] = np.exp(x[:, n]) / S
        return result
    else:
        print("The input array is not 1- or 2-dimensional.")
'''
def getMin(a,b,c):
    max=0
    if a>b:
        max=b
    else:
        max=a
    if max>c:
        return c
    else:
        return max


# Evaluate
def evaluate(qf1, qf2, qv1, qv2, ql, qc1, qc2, gf, gv, gl, gc):
    query1 = qf1.view(-1, 1)
    query2 = qf2.view(-1, 1)
    #query = (query1+query2+query3)/3
    #query_1 = qf1.view(1, -1)
    query_2 = qf2.view(1, -1)
    dist_12 = torch.mm(query_2, query1) + 1.
    query_v1 = qv1.view(-1, 1)
    query_v2 = qv2.view(-1, 1)
    # print(query.shape)
    score1 = torch.mm(gf, query1) + 1.
    score1 = score1.squeeze(1).cpu()
    view1 = torch.mm(gv, query_v1) + 1.
    view1 = view1.squeeze(1).cpu()
    view1 = view1.numpy()
    score1 = score1.numpy()
    score2 = torch.mm(gf, query2) + 1.
    score2 = score2.squeeze(1).cpu()
    view2 = torch.mm(gv, query_v2) + 1.
    view2 = view2.squeeze(1).cpu()
    view2 = view2.numpy()
    score2 = score2.numpy()
    score3 = (score1+score2)/2
    view = np.concatenate((view1, view2), axis=0)
    view = view.reshape(2, len(view1))
    view_soft = softmax(view)

    index_max = np.argmax(view_soft, axis=0)  # 得到每一列最大值索引
    score_2 = score1 * view_soft[0] + score2 * view_soft[1]
    for i, value in enumerate(index_max):
        if value == 0:
            score1[i] = 1.1 * score1[i]
        elif value == 1:
            score2[i] = 1.1 * score2[i]

    score_1 = (score1 + score2)/2
    score = score_1*0.4 + score_2*0.3 + score3*0.3  #0.65 0.45
    #score = score_1
    '''
    for i, value in enumerate(index_max):
        if value == 0 and qc1 != 1000:
            if score1[i] > score2[i] + score3[i]:
                score1[i] = 1.2 * score1[i]

        elif value == 1 and qc2 != 1000:
            if score2[i] > score1[i] + score3[i]:
                score2[i] = 1.2 * score2[i]

        elif value == 2 and qc3 != 1000:
            if score3[i] > score2[i] + score1[i]:
                score3[i] = 1.2 * score3[i]
            #else:
            #    score3[i] = 0.785 * score3[i]
    '''
    '''
    for i, value in enumerate(index_max):
        #if(j<1000):
        if value == 0:
            if score1[i] > lim:
                #score[i] = score1[i] * view[0][i] + score2[i] * view[1][i] + score3[i] * view[2][i]
                score1[i] = score1[i] * 1.2
            else:
                #score1[i] = 10 * score1[i]
                #score[i] = (score1[i]*0.8 + score2[i] + score3[i])/3
                score1[i] = score1[i] * 0.9
                #score2[i] = score2[i] * view[1][i]
                #score3[i] = score3[i] * view[2][i]
        elif value == 1:
            if score2[i] > lim:
                #score[i] = score1[i] * view[0][i] + score2[i] * view[1][i] + score3[i] * view[2][i]
                score2[i] = score2[i] * 1.2
            else:
                #score[i] = (score1[i]*1.2 + score2[i] + score3[i])/3
                score2[i] = score2[i] * 0.8
                #score2[i] = score2[i] * view[1][i]
                #score3[i] = score3[i] * view[2][i]
        elif value == 2:
            if score3[i] > lim:
                #score[i] = score1[i] * view[0][i] + score2[i] * view[1][i] + score3[i] * view[2][i]
                score3[i] = score3[i] * 1.2
            else:
                #score[i] = (score1[i]*1.2 + score2[i] + score3[i])/3
                score3[i] = score3[i] * 0.9
                #score2[i] = score2[i] * view[1][i]
                #score3[i] = score3[i] * view[2][i]

    score = score1 * view[0] + score2 * view[1] + score3 * view[2]
    #score = (score1 + score2 + score3) / 3
    '''
    '''
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
    '''
    #pdb.set_trace()
    #score = score1 * view[0] + score2 * view[1] + score3 * view[2]
    #score = (score1+score2)/2
    index = np.argsort(score)  # from small to large array([ 6692,  3951, 12288, ...,   144,  2906,   659])
    index = index[::-1]  # array([  659,  2906,   144, ..., 12288,  3951,  6692]) from large to small array
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere((gc == qc1) | (gc == qc2))
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # good_index = query_index
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())
    # junk_index = junk_index1

    CMC_tmp = compute_mAP(index, good_index, junk_index, gc, gf)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index, gc, gf):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    inp = 0
    cp = 0
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc, inp, cp

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index_map = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index_map, good_index)

    mask_cam = copy.deepcopy(mask)
    cam = index_map[mask]
    for i in range(len(cam) - 1):
        x = gc[cam[i]]
        for k in range(i + 1, len(cam)):
            y = gc[cam[k]]
            simx_y = torch.mm(gf[cam[k]].view(1, -1), gf[cam[i]].view(-1, 1))
            if x == y and simx_y > 0.6:
                mask_cam[np.argwhere(index_map == cam[k])] = False

    rows_good = np.argwhere(mask == True)
    max_rows_good = np.max(rows_good)
    #max_rows_good = rows_good[-2][0]
    inp = ngood / (max_rows_good + 1.0)

    rows_good = rows_good.flatten()

    rows_good_cam = np.argwhere(mask_cam == True)
    rows_good_cam = rows_good_cam.flatten()

    cmc[rows_good[0]:] = 1

    for i in range(len(rows_good_cam)):
        d_recall_cam = 1.0 / len(rows_good_cam)
        precision_cam = (i + 1) * 1.0 / (rows_good_cam[i] + 1)
        cp = cp + d_recall_cam * precision_cam

    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc, inp, cp


######################################################################
# result_query = scipy.io.loadmat('pytorch_result_query.mat')
result_gallery = scipy.io.loadmat('pytorch_result_gallery_151_mmtm.mat')
# result = scipy.io.loadmat('pytorch_result.mat')
result = scipy.io.loadmat('pytorch_result_resnet_151_mmtm_miss.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_view = torch.FloatTensor(result['query_v'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result_gallery['gallery_f'])
gallery_view = torch.FloatTensor(result_gallery['gallery_v'])
gallery_cam = result_gallery['gallery_cam'][0]
gallery_label = result_gallery['gallery_label'][0]
#spdb.set_trace()
print(query_feature.shape)
print(query_label)
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
inp = 0.0
cp = 0.0
# print(query_label)
for i in range(0, len(query_label), 2):
    #ap_tmp, CMC_tmp, inp_tmp = evaluate(query_feature[i], query_feature[i + 1], query_feature[i + 2], query_label[i],query_cam[i], query_cam[i + 1], query_cam[i + 2], gallery_feature, gallery_label, gallery_cam)

    #if i%2==0:
    ap_tmp, CMC_tmp, inp_tmp, cp_tmp = evaluate(query_feature[i],query_feature[i+1], query_view[i],query_view[i+1],query_label[i], query_cam[i], query_cam[i+1], gallery_feature, gallery_view,gallery_label,
                           gallery_cam)
    '''
    elif i%2==1:
        ap_tmp, CMC_tmp, inp_tmp = evaluate(query_feature[i-1], query_feature[i], query_view[i-1],query_view[i],query_label[i],
                                   query_cam[i], query_cam[i-1], gallery_feature, gallery_view,gallery_label,
                                  gallery_cam)
    
    else:
        ap_tmp, CMC_tmp, inp_tmp = evaluate(query_feature[i - 1], query_feature[i-2], query_feature[i], query_view[i-1],query_view[i-2],query_view[i],query_label[i],
                                   query_cam[i], query_cam[i-1], query_cam[i-2], gallery_feature, gallery_view, gallery_label,
                                   gallery_cam)
    '''
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    cp += cp_tmp
    inp += inp_tmp
    # print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC / len(query_label)*2 # average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mCP:%f mAP:%f mINP:%f' % (CMC[0], CMC[4], CMC[9], cp / len(query_label)*2, ap / len(query_label)*2, inp/len(query_label)*2))
