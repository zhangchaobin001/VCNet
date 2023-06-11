import torch
from collections import Counter, defaultdict
import pdb
import numpy as np
import os
file_path = "../mqveri_gan/rename"
'''
view_miss = defaultdict(str)
view_miss[0] = 1
view_miss[2] = 5
if 2 not in view_miss.values() and 5 not in view_miss.values():
    print("miss ce")
elif 3 not in view_miss.values() and 6 not in view_miss.values() and 0 not in view_miss.values():
    for item, value in view_miss.items():
        if value == 2 or value == 5:
            print(item, type(item))
    print("miss qian")
elif 1 not in view_miss.values() and 4 not in view_miss.values() and 7 not in view_miss.values():
    print("miss hou")
'''
'''
gc = np.array([0,0,0,1,1,0,2,1,4])
index = np.array([1,2,3,5,7])
mask = np.in1d(index, index, invert=False)
for i in range(len(index)-1):
    #pdb.set_trace()
    x = gc[index[i]]
    for k in range(i+1, len(index)):
        y = gc[index[k]]
        if x == y:
            mask[k] = False
print(mask)
print(index[mask])
'''
# gpu = "0,1"
# gpus = ''.join(gpu.split())
# gids = [int(gid) for gid in gpus.split(',')]
# print(gids)
# for root, dirs, files in os.walk(file_path, topdown=True):
#     for dir in dirs:
'''
        #添加摄像头id信息
        rename = file.split(".")[0] + "_" + '1.jpg'
        NewName = os.path.join(root, rename)
        OldName = os.path.join(root, file)
        os.rename(OldName, NewName)

        #给图片添加id信息
        rename = root.split("/")[-1] + "_" + file
        NewName = os.path.join(root, rename)
        OldName = os.path.join(root, file)
        os.rename(OldName, NewName)
        

        
        #重命名摄像头id,以后应该用不到了
        camid = int(file.split('.')[0][-1])
        #import pdb
        #pdb.set_trace()
        if camid == 1:
            rename = file.split('.')[0] + "_" + "0.jpg"
        elif camid == 0:
            rename = file.split('.')[0] + "_" + "2.jpg"
        elif camid == 2:
            rename = file.split('.')[0] + "_" + "1.jpg"
        NewName = os.path.join(root, rename)
        OldName = os.path.join(root, file)
        os.rename(OldName, NewName)
        
        #将命名好摄像头id的图片移动回去
        #pdb.set_trace()
        #newdir = int(dir) + 199
        #NewName = os.path.join(root, dir + str(199))
        #OldName = os.path.join(root, dir)
        # os.rename(os.path.join(root, dir), os.path.join(root, str(199)+dir)
'''
a = torch.ones([2,2])
b = torch.ones([2,2])
c = torch.zeros([2,2])
d = (a + b + c) / 3
print(a,d,d.shape)


