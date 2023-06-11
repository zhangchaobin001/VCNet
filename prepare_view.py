import os
from shutil import copyfile
import pdb
import io

# You only need to change this line to your dataset download path
download_path = "/data/server77_data_a/public_data/vehicleReid/Veri776"
path = './keypoint_test.txt'
if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = "../veri/view"
if not os.path.isdir(save_path):
    os.mkdir(save_path)
#---------------------------------------
#train_all

f2 = io.open(path, 'r', encoding= 'UTF-8')
train_path = download_path + '/image_test'
train_save_path = save_path + '/test'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
for i in f2:
    filename = i.split('/')[-1].split(' ')[0]   #0190_c014_00051260_1.jpg
    label = i.split('/')[-1][0:4]  #0190
    view = i[-2]  #6
    src_path = train_path + '/' + filename  #../veri/image_train/0190_c014_00051260_1.jpg
    if not os.path.exists(src_path):
        continue
    dst_path = train_save_path + '/' + label #../veri/view/test/0190
    new_filename = filename.split('.')[0]+'_'+view+'.jpg'  #0190_c014_00051260_1_6.jpg
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    copyfile(src_path, dst_path + '/' + new_filename)
f2.close()

pdb.set_trace()

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
#-----------------------------------------
#query
query_path = download_path + '/image_query'
query_save_path = download_path + '/pytorch/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = query_path + '/' + name
        dst_path = query_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
#-----------------------------------------
#multi-query
query_path = download_path + '/multi_query'
# for dukemtmc-reid, we do not need multi-query
if os.path.isdir(query_path):
    query_save_path = download_path + '/pytorch/multi-query'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = query_path + '/' + name
            dst_path = query_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

#-----------------------------------------
#gallery
gallery_path = download_path + '/image_test'
gallery_save_path = download_path + '/pytorch/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = gallery_path + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)




#---------------------------------------
#train_val
train_path = download_path + '/image_train'
train_save_path = download_path + '/pytorch/train'
val_save_path = download_path + '/pytorch/val'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            dst_path = val_save_path + '/' + ID[0]  #first image is used as val image
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
