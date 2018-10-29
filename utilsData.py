import torch

import os
import os.path
import numpy as np
import random
import torch
import cv2
import math
import glob
import scipy.misc
import torch.utils.data as udata
import torch.nn as nn

from torch.autograd import Variable
from PIL import Image
from os import listdir
from os.path import join
from torch.utils.data.dataset import Dataset
from skimage.measure.simple_metrics import compare_psnr
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from utils import *

# 2 class(es) + 1 function(s)
class DatasetImage(Dataset):
    def __init__(self, *paths, **kwargs):
        super(Dataset, self).__init__()
        self.imageSets = []
        self.mode = kwargs['mode']
        for arg in paths:
            if self.mode == 'train':
                imgArray = dataPrepare(arg, kwargs['augTimes'], kwargs['cropPatch'], kwargs['cropSize'], kwargs['stride'])
            else:
                imgArray = dataPrepare(arg, 1, False)#, cropSize, stride)
            imgArray = np.transpose(imgArray, (0, 3,1,2))
            self.imgArray = np.expand_dims(imgArray, axis=1)
            self.imageSets.append(self.imgArray)
        self.imageSets = np.concatenate(tuple(self.imageSets), axis=1)
    def __getitem__(self, index):
        img = self.imageSets[index]
        return torch.Tensor(img)

    def __len__(self):
        return self.imgArray.shape[0]


class DatasetImageWithNoise(Dataset):
    def __init__(self, path, **kwargs):
        super(Dataset, self).__init__()
        self.imageSets = []
        self.mode = kwargs['mode']
        self.numOfNoisySets = kwargs['numOfNoisySets']
        for num in range(self.numOfNoisySets + 1):
            if self.mode == 'train':
                imgArray = dataPrepare(''.join([path, '/train']), kwargs['augTimes'],
                 kwargs['cropPatch'], kwargs['cropSize'], kwargs['stride'], num>0, kwargs['noiseLevel'])
            else:
                imgArray = dataPrepare(''.join([path, '/val']), 1, True, kwargs['cropSize'], kwargs['stride'], num>0, kwargs['noiseLevel'])
               
            imgArray = np.transpose(imgArray, (0, 3,1,2))
            self.imgArray = np.expand_dims(imgArray, axis=1)
            self.imageSets.append(self.imgArray)
        self.imageSets = np.concatenate(tuple(self.imageSets), axis=1)
    def __getitem__(self, index):
        img = self.imageSets[index]
        return torch.Tensor(img)

    def __len__(self):
        return self.imgArray.shape[0]


def dataPrepare(dataPath, augTimes=1, cropPatch=False, cropSize=30, stride=5, withNoise=True, noiseLevel=15, inhomoDistribute=False):
    # train
    files = completePathImportFile(dataPath)
    dataLength = augTimes * len(files)

    if cropPatch == False:        
        h, w, _ = cv2.imread(files[0]).shape
        sz = h
    else:
        scales = [1, 0.9, 0.8, 0.7]
        sz = cropSize
        for i in range(len(files)):
            lenPatch = patchCount(cv2.imread(files[i]), scales=scales, cropSize=cropSize, stride=stride)
            dataLength += lenPatch - 1
       
    imgArray = np.empty(shape=(dataLength, sz, sz, 3))
    
    sampleIndex = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])        
        img = np.float32(normalize(img))

        if withNoise:
            img = addNoise(img, noiseLevel, ifInhomogeneous=inhomoDistribute)
        
        if cropPatch:  
            img = patchCrop(img, scales, sz, stride)
        else:
            img = np.expand_dims(img.copy(), 3)
        
        for n in range(img.shape[3]):
            imgArray[sampleIndex] = img[:,:,:,n].copy()

            sampleIndex += 1
            for m in range(augTimes-1):
                rand = np.random.randint(1,8)
                imgArray[sampleIndex] = dataAug(img[:,:,:,n].copy(), rand)
                sampleIndex += 1

    return imgArray