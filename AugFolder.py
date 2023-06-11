"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from torchvision import datasets
import os
import numpy as np
import random
import pdb


class AugFolder(datasets.ImageFolder):

    def __init__(self, root, transform, transform2):
        super(AugFolder, self).__init__(root, transform, transform2)
        self.transform2 = transform2

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample2 = sample.copy() 
        if self.transform is not None:
            sample = self.transform(sample)
            sample2 = self.transform2(sample2)
        return sample, sample2, target

class ViewAugFolder(datasets.ImageFolder):

    def __init__(self, root, transform, transform2):
        super(ViewAugFolder, self).__init__(root, transform, transform2)
        self.transform2 = transform2

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample2 = sample.copy()
        if self.transform is not None:
            sample = self.transform(sample)
            sample2 = self.transform2(sample2)
        filename = os.path.basename(path)
        view = filename.split('_')[-1].split(".")[0]
        view = int(view)
        return sample, sample2, view

class ViewFolder(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(ViewFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        filename = os.path.basename(path)
        view = filename.split('_')[-1].split(".")[0]
        view = int(view)
        return sample, view

class ViewFolder_mv(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(ViewFolder_mv, self).__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        filename = os.path.basename(path)

        return sample, filename, path

class MyFolder(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(MyFolder, self).__init__(root, transform)
        #pdb.set_trace()

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        filename = os.path.basename(path)
        #print(path,target)
        #pdb.set_trace()
        #要想在这里进行调试，numwork需要设置为0
        label = int(filename.split("_")[0])
        camera = filename.split("_")[-1].split(".")[0]
        camera_id = int(camera)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, label, camera_id