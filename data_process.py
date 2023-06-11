import os
from shutil import copyfile
import pdb

bianhao = set()
with open("/data/server77_data_c/public_data/wait_for_label/完成编号1.txt", 'r', encoding='utf-8') as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip('\n')
        try:
            key = line.split('\\')[-2]
        except:
            print(1)
        value = line.split('\\')[-1]
        bianhao.add(value)
pdb.set_trace()
path = "/data/server77_data_c/public_data/wait_for_label/mqveri"
rootPath = "/data/server77_data_c/public_data/wait_for_label/image"
'''
for i in bianhao:
    filepath = os.path.join(path, i)
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
pdb.set_trace()

for root, dirs, files in os.walk(rootPath, topdown=True):
    #for i, file in enumerate(files):
    if files in bianhao:

    for dir in dirs:
        for img in bianhao:
            fileName = img.split('\\')[-1]
            folder = img.split('\\')[-2]
            camid = img.split('_')[-1][0]
            '''
'''
            if camid == 0:
                find = 'bei'
            elif camid == 1:
                find = 'dong'
            elif camid == 2:
                find = 'xi'
            elif camid == 3:
                find = '高新区云飞路文曲路北'
            elif camid == 4:
                find = '高新区云飞路文曲路南'
            elif camid == 5:
                find = '高新区云飞路文曲路西'
            elif camid == 6:
                find = '高新区云飞路永和路北'
            elif camid == 7:
                find = '高新区云飞路永和路南'
            elif camid == 8:
                find = '高新区云飞路永和路西'
            elif camid == 9:
                find = '望江西路文曲路北'
            elif camid == 10:
                find = '望江西路文曲路东'
            elif camid == 11:
                find = '望江西路文曲路西'
'''
'''
            p = plate.split("_")[0]
            id = plate.split("_")[-1]
            if(p == dir):
                NewName = os.path.join(root, dir)
                OldName = os.path.join(root, dir+"_"+id)
                os.rename(NewName, OldName)
'''