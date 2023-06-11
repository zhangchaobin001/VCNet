# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

from collections import defaultdict
import numpy as np
import copy
import random
import pdb
import torch
from tqdm import tqdm
import pdb
from torch.utils.data.sampler import Sampler

class RandomIdentity(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)   # 定义一个自动扩充的dictionary
        for index, (_, pid) in enumerate(tqdm(self.data_source.imgs)):
            self.index_dic[pid].append(index)    #按照id归类,[pid,index]
        self.pids = list(self.index_dic.keys())
        self.num_indentities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_indentities)    #对每一个id作shuffle，但保证每个id下的图片不变
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]  #当前id下的所有图片序号
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace) #从每个id下从t中随机选择num_instances张图片，不重复
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_instances * self.num_indentities

class RandomIdentityGan(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)   # 定义一个自动扩充的dictionary
        for index, (_, pid) in enumerate(tqdm(self.data_source.imgs)):
            self.index_dic[pid].append(index)    #按照id归类,[pid,index]
        self.pids = list(self.index_dic.keys())
        self.num_indentities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_indentities)    #对每一个id作shuffle，但保证每个id下的图片不变
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]  #当前id下的所有图片序号
            #replace = False if len(t) >= self.num_instances else True
            if len(t) < self.num_instances:
                re_select = np.random.choice(t, size=self.num_instances-len(t), replace=True)
                t.extend(re_select.tolist()) #从每个id下从t中随机选择num_instances张图片，不重复
            else:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_instances * self.num_indentities


#训练视角生成器用的采样方式
class RandomIdentitySamplerGan(Sampler):

    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (Dataset): dataset to sample from.
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source #传过来的就是image_datasets['train']
        self.batch_size = batch_size #36
        self.num_instances = num_instances #3
        self.num_pids_per_batch = self.batch_size // self.num_instances  # 12
        self.index_dic = defaultdict(list) #
        self.cmaid_dic = defaultdict(str) #存储camid
        for index, (filepath, pid) in enumerate(self.data_source.imgs):
            camid = filepath.split("_")[-1].split(".")[0]
            self.cmaid_dic[index] = camid  #
            self.index_dic[pid].append(index)  # 以字典形式存储，key表示pid,value表示图片的索引index
        self.pids = list(self.index_dic.keys())  # 存储pid[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10，...]

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list) #将同一个id的图片每3张放一起，字典形式
        #在这里就需要进行修改了，将同一个id下不同摄像头id的图片每3张放一起
        #print(self.data_source.classes)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])  # 每个id下的图片索引[0,1,2,3,...,53]
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)  # 如果图片不够，随机重复取
            random.shuffle(idxs)  # 将图片打乱
            batch_idxs = [] #暂存一个batch里面同一个id的图片
            camid = defaultdict(list)
            cam_id = set()
            for idx in idxs:
                cid = self.cmaid_dic[idx]
                cam_id.add(cid)
                camid[cid].append(idx)
            #pdb.set_trace()
            #print(self.data_source.imgs)
            #print(pid)
            for i in range(len(idxs)//self.num_instances):
                for cid in cam_id:
                    image = random.sample(camid[cid], 1)
                    batch_idxs.append(image[0])
                    camid[cid].remove(image[0])
                    if len(batch_idxs) == self.num_instances: #判断同一个id下图片的数量是否达到num_instances
                        batch_idxs_dict[pid].append(batch_idxs)  # {0:[[40,11,33],[14,42,13],..., 1:[]}
                        batch_idxs = []
            camid.clear()
            cam_id.clear()
        avai_pids = copy.deepcopy(self.pids)  # [0,1,2,3,...,575]
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch: #判断当id够用的时候
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)  # 随机选择self.num_pids_per_batch数量的id
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)  # [40,11,33] 对每一个选择的id从中选取3张图片
                final_idxs.extend(batch_idxs)  # [40,11,33,14,42,13,...]
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)  # 没有图片就删除id


        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (Dataset): dataset to sample from.
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances  # 12
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(self.data_source.imgs):
            self.index_dic[pid].append(index)  # pid��Ϊkey�����ͼƬ���
        self.pids = list(self.index_dic.keys())  # һ���������pid��list

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])  # [0,1,2,3,...,53]
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)  # ��idxs����ѡ3��ͼ���������ظ�
            random.shuffle(idxs)  # �����id�е�ͼƬ����
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)  # {0:[[40,11,33],[14,42,13],..., 1:[]}
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)  # [0,1,2,3,...,575]
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)  # ���ѡ��12��pid
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)  # [40,11,33]
                final_idxs.extend(batch_idxs)  # [40,11,33,14,42,13,...]
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)  # ȡ���˾�ɾ��

        return iter(final_idxs)

    def __len__(self):
        return self.length