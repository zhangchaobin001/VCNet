import shutil
import os

path = '/data/cbzhang4/mqveri_gan/'
qian = 0
hou = 0
ce = 0
#1hou 2ce 0qian
for root, dirs, files in os.walk(path, topdown=True):
    for i, file in enumerate(files):
        #重命名摄像头id,以后应该用不到了
        try:
            camid = int(file.split('.')[0][-1])
            if(camid == 0):
                qian  = qian + 1
            elif(camid == 1):
                hou = hou + 1
            elif(camid == 2):
                ce = ce + 1
        except:
            print(file)
print(qian, hou, ce)
        #import pdb
        #pdb.set_trace()
        #if camid != 0:
        #    os.remove(os.path.join(root, file))