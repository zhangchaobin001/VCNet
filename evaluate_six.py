# -*- coding: utf-8 -*-

import scipy.io
import torch
import numpy as np
# import time
import os
import copy
import pickle
import pdb
import sys
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
def evaluate(qf1, qf2, qf3,qf4, qf5, qf6, qv1, qv2, qv3, qv4, qv5, qv6, ql, qc1, qc2, qc3, qc4, qc5, qc6, gf, gv, gl, gc):
    query1 = qf1.view(-1, 1)
    query2 = qf2.view(-1, 1)
    query3 = qf3.view(-1, 1)
    query4 = qf4.view(-1, 1)
    query5 = qf5.view(-1, 1)
    query6 = qf6.view(-1, 1)
    #query = (query2+query3+query1+query4+query5+query6)/6
    #query = (query1+query2+query3)/3
    #query_1 = qf1.view(1, -1)
    #query_2 = qf2.view(1, -1)
    #query_3 = qf3.view(1, -1)
    query_v1 = qv1.view(-1, 1)
    query_v2 = qv2.view(-1, 1)
    query_v3 = qv3.view(-1, 1)
    query_v4 = qv4.view(-1, 1)
    query_v5 = qv5.view(-1, 1)
    query_v6 = qv6.view(-1, 1)
    # print(query.shape)
    score1 = torch.mm(gf, query1) + 1.
    score1 = score1.squeeze(1).cpu()
    view1 = torch.mm(gv, query_v1) + 1.
    view1 = view1.squeeze(1).cpu()
    view1 = view1.numpy()
    score1 = score1.numpy()
    # score1 = score1.numpy().reshape(1,-1)
    # score1 = preprocessing.normalize(score1, norm='l2').squeeze()
    score2 = torch.mm(gf, query2) + 1.
    score2 = score2.squeeze(1).cpu()
    view2 = torch.mm(gv, query_v2) + 1.
    view2 = view2.squeeze(1).cpu()
    view2 = view2.numpy()
    score2 = score2.numpy()
    # score2 = score2.numpy().reshape(1,-1)
    # score2 = preprocessing.normalize(score2, norm='l2').squeeze()
    score3 = torch.mm(gf, query3) + 1.
    score3 = score3.squeeze(1).cpu()
    view3 = torch.mm(gv, query_v3) + 1.
    view3 = view3.squeeze(1).cpu()
    score3 = score3.numpy()
    view3 = view3.numpy()

    score4 = torch.mm(gf, query4) + 1.
    score4 = score4.squeeze(1).cpu()
    view4 = torch.mm(gv, query_v4) + 1.
    view4 = view4.squeeze(1).cpu()
    score4 = score4.numpy()
    view4 = view4.numpy()

    score5 = torch.mm(gf, query5) + 1.
    score5 = score5.squeeze(1).cpu()
    view5 = torch.mm(gv, query_v5) + 1.
    view5 = view5.squeeze(1).cpu()
    score5 = score5.numpy()
    view5 = view5.numpy()

    score6 = torch.mm(gf, query6) + 1.
    score6 = score6.squeeze(1).cpu()
    view6 = torch.mm(gv, query_v6) + 1.
    view6 = view6.squeeze(1).cpu()
    score6 = score6.numpy()
    view6 = view6.numpy()

    # score3 = score3.numpy().reshape(1,-1)
    # score3 = preprocessing.normalize(score3, norm='l2').squeeze()
    # 计算视角相似度
    '''
    view1 = softmax(view1)
    view2 = softmax(view2)
    view3 = softmax(view3)
    '''
    view = np.concatenate((view1, view2, view3, view4, view5, view6), axis=0)
    view = view.reshape(6, len(view1))
    view = softmax(view)
    #print(view)
    #score = torch.mm(gf, query)
    #score = score.squeeze(1).cpu()
    #score = score.numpy()
    score_4 = (score1 + score2 + score3 + score4 + score5 + score6) / 6
    score_2 = score1 * view[0] + score2 * view[1] + score3 * view[2] + score4 * view[3] + score5 * view[4] + score6 * view[5]

    index_max = np.argmax(view, axis=0)  # 得到每一列最大值索引
    #pdb.set_trace()
    for i, value in enumerate(index_max):
        if value == 0:
            score1[i] = 1.2 * score1[i]

        elif value == 1:
            score2[i] = 1.2 * score2[i]

        elif value == 2:
            score3[i] = 1.2 * score3[i]

        elif value == 3:
            score4[i] = 1.2 * score4[i]

        elif value == 4:
            score5[i] = 1.2 * score5[i]

        elif value == 5:
            score6[i] = 1.2 * score6[i]

    score_1 = (score1 + score2 + score3 + score4 + score5 + score6)/6
    #score = score_1*0.8 + score_2*0.2  #0.6 0.4
    score = score_4
    index = np.argsort(score)  # from small to large array([ 6692,  3951, 12288, ...,   144,  2906,   659])
    index = index[::-1]  # array([  659,  2906,   144, ..., 12288,  3951,  6692]) from large to small array
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere((gc == qc1) | (gc == qc2) | (gc == qc3))
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # good_index = query_index
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())
    # junk_index = junk_index1

    CMC_tmp = compute_mAP(index, good_index, junk_index, gc, gv)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index, gc, gv):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    inp = 0
    cp = 0
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc, inp, cp
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)  #测试一维数组的每个元素是否也存在于第二个数组中。
    index_map = index[mask]
    #里面如果有多余的重复元素：

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index_map, good_index)

    mask_cam = copy.deepcopy(mask)
    cam = index_map[mask]
    # mask_cam = np.in1d(cam, cam, invert=False)
    for i in range(len(cam) - 1):
        x = gc[cam[i]]
        for k in range(i + 1, len(cam)):
            y = gc[cam[k]]
            simx_y = torch.mm(gv[cam[k]].view(1, -1), gv[cam[i]].view(-1, 1))
            if x == y and simx_y > 0.6:
                mask_cam[np.argwhere(index_map == cam[k])] = False

    rows_good = np.argwhere(mask == True)
    rows_good_cam = np.argwhere(mask_cam == True)
    max_rows_good = np.max(rows_good)
    inp = ngood / (max_rows_good + 1.0)

    rows_good = rows_good.flatten()
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
result_gallery = scipy.io.loadmat('pytorch_result_gallery_ft.mat')
result = scipy.io.loadmat('pytorch_result_query_ft.mat')
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
CMC = torch.IntTensor(len(gallery_label)).zero_()
CP = 0.0
ap = 0.0
inp = 0.0

# print(query_label)
for i in range(0, len(query_label), 6):
    #ap_tmp, CMC_tmp, inp_tmp = evaluate(query_feature[i], query_feature[i + 1], query_feature[i + 2], query_label[i],query_cam[i], query_cam[i + 1], query_cam[i + 2], gallery_feature, gallery_label, gallery_cam)
    #pdb.set_trace()
    #if i%3==0:
    ap_tmp, CMC_tmp, inp_tmp, cp_tmp = evaluate(query_feature[i], query_feature[i + 1], query_feature[i + 2],query_feature[i+3], query_feature[i + 4], query_feature[i + 5],
                                                query_view[i],query_view[i + 1], query_view[i + 2],query_view[i + 3], query_view[i + 4],query_view[i + 5],
                                                query_label[i], query_cam[i], query_cam[i + 1], query_cam[i + 2], query_cam[i + 3], query_cam[i + 4], query_cam[i + 5],
                                                gallery_feature, gallery_view, gallery_label, gallery_cam)


    '''
    if query_cam[i] == 1000:
        continue
    elif i%3==0 and query_cam[i] != 1000:
        ap_tmp, CMC_tmp, inp_tmp = evaluate(query_feature[i],query_feature[i+1],query_feature[i+2], query_view[i],query_view[i+1],query_view[i+2],query_label[i], query_cam[i], query_cam[i+1], query_cam[i+2], gallery_feature, gallery_view,gallery_label,
                               gallery_cam)
    elif i%3==1 and query_cam[i] != 1000:
        ap_tmp, CMC_tmp, inp_tmp = evaluate(query_feature[i-1], query_feature[i + 1], query_feature[i], query_view[i-1],query_view[i+1],query_view[i],query_label[i],
                                   query_cam[i], query_cam[i-1], query_cam[i+1], gallery_feature, gallery_view,gallery_label,
                                   gallery_cam)
    elif i%3==2 and query_cam[i] != 1000:
        ap_tmp, CMC_tmp, inp_tmp = evaluate(query_feature[i - 1], query_feature[i-2], query_feature[i], query_view[i-1],query_view[i-2],query_view[i],query_label[i],
                                   query_cam[i], query_cam[i-1], query_cam[i-2], gallery_feature, gallery_view, gallery_label,
                                   gallery_cam)
    '''
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    CP = CP + cp_tmp
    inp += inp_tmp
    # print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC / len(query_label) *6 #average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f MCP:%f mAP:%f mINP:%f' % (CMC[0], CMC[4], CMC[9], CP/ len(query_label) *6, ap / len(query_label) *6, inp/ len(query_label) *6))
