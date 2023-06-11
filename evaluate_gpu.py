import scipy.io
import torch
import numpy as np
import pdb
import copy
import sys
np.set_printoptions(threshold=sys.maxsize)
#import time
import os
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#######################################################################
# Evaluate
def evaluate(score,ql,qc,gl,gc,gf,qv,gv):
    #print(score.shape)
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    #index = index[0:100]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    if qc == -1: #VeID has no camera ID
        camera_index = []
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())

    #print(good_index)    
    CMC_tmp = compute_mAP(index, good_index, junk_index, gc,gf,qv,gv)
    return CMC_tmp

def compute_mAP(index, good_index, junk_index, gc,gf,qv,gv):
    ap = 0
    cp = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc,cp

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index_map = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index_map, good_index)
    #pdb.set_trace()
    #去除同摄像头下的正样本
    mask_cam = copy.deepcopy(mask)
    cam = index_map[mask]
    for i in range(len(cam) - 1):
        x = gc[cam[i]]
        for k in range(i + 1, len(cam)):
            y = gc[cam[k]]
            simx_y = torch.mm(gv[cam[k]].view(1,-1),gv[cam[i]].view(-1,1))
            if x == y and simx_y > 0.7:
                mask_cam[np.argwhere(index_map == cam[k])] = False

    rows_good = np.argwhere(mask==True) #正样本的index
    max_rows_good = np.max(rows_good)
    inp = ngood/(max_rows_good+1.0)
    rows_good = rows_good.flatten()

    rows_good_cam = np.argwhere(mask_cam == True)
    rows_good_cam = rows_good_cam.flatten()

    cmc[rows_good[0]:] = 1

    for i in range(len(rows_good_cam)):
        d_recall_cam = 1.0 / len(rows_good_cam)
        precision_cam = (i + 1) * 1.0 / (rows_good_cam[i] + 1)
        cp = cp + d_recall_cam * precision_cam

    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc, inp, cp

######################################################################
def calculate_result(gallery_feature, gallery_label, gallery_cam, query_feature, query_label, query_cam, query_view, gallery_view, result_file, pre_compute_score=None):
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    inp = 0.0
    cp = 0.0
    if pre_compute_score is None:
        score = torch.mm(query_feature, torch.transpose(gallery_feature,0,1))
        score = score.cpu().numpy()
    else: 
        score = pre_compute_score

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp, inp_tmp, cp_tmp = evaluate(score[i],query_label[i],query_cam[i],gallery_label,gallery_cam,gallery_feature, query_view, gallery_view)
        if CMC_tmp[0]==-1:
            continue
        # if CMC_tmp[0]==0:
        #     print(i)
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        cp += cp_tmp
        inp += inp_tmp

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    str_result = 'Rank@1:%f Rank@5:%f Rank@10:%f mCP:%f mAP:%f mINP:%f\n'%(CMC[0], CMC[4], CMC[9], cp/len(query_label), ap/len(query_label), inp/len(query_label))
    print(str_result)
    text_file = open(result_file, "a")
    text_file.write(str_result)
    text_file.close()
    return score

if __name__ == '__main__':
    result = scipy.io.loadmat('pytorch_result_query_ft.mat')
    result_gallery = scipy.io.loadmat('pytorch_result_gallery_ft.mat')
    query_feature = torch.FloatTensor(result['query_f'])[:, 0:512]
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    query_view = torch.FloatTensor(result['query_v'])
    gallery_feature = torch.FloatTensor(result_gallery['gallery_f'])[:, 0:512]
    gallery_cam = result_gallery['gallery_cam'][0]
    gallery_label = result_gallery['gallery_label'][0]
    gallery_view = torch.FloatTensor(result_gallery['gallery_v'])
    print(query_feature.shape)
    #print(np.unique(query_cam))
    calculate_result(gallery_feature, gallery_label, gallery_cam, query_feature, query_label, query_cam,query_view, gallery_view, 'tmp.txt')
