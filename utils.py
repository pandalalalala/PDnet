import torch

import os
import os.path
import numpy as np
import random
import torch
import cv2
import glob
import math
import scipy.misc
import torch.utils.data as udata
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from os import listdir
from os.path import join
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from skimage.measure.simple_metrics import compare_psnr
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

# 11 function(s) + 3 noise function(s) + 1 validatate function(s)

def val(model, dataPath, epoch, noiseLevel=12, mode='pure_denoise', ifInhomogeneous=False):
    model.eval()

    # load data info
    print('Evaluating ...\n')
    imageFiles = completePathImportFile(dataPath)

    # process data
    psnr_predict, psnr_input = 0, 0
    for f in range(len(imageFiles)):
        # image
        #imgClear = cv2.imread(imageFiles[f])
        imgClear = Image.open(imageFiles[f])
        imgInput = imgClear.copy()
        if mode ==  'sr':
            h, w = imgClear.size
            imgInput = imgInput.resize((w/4, h/4), Image.NEAREST)
        imgClear, imgInput = np.array(imgClear), np.array(imgInput)

        if len(imgClear.shape) == 2:
            imgClear, imgInput = np.expand_dims(imgClear, axis=2), np.expand_dims(imgInput, axis=2)
        imgInput = addNoise(imgInput, sigma=noiseLevel*255, ifInhomogeneous=ifInhomogeneous, mu=0)
        cv2.imwrite('logimg/input%d.png' %f, imgInput)

        imgClear = tensorVariableNumpyImage(imgClear)
        imgInput = tensorVariableNumpyImage(imgInput)

        with torch.no_grad(): # this can save much memory
            outRes = model(imgInput)
            outRes = sum(outRes)

        psnr_input += batchPSNR(imgInput, imgClear, 1.)
        psnr_predict += batchPSNR(imgInput - outRes, imgClear, 1.)
        imgSaveTensor(imgInput, 'logimg', 'output%d.png' %f)

    psnr_predict /= len(imageFiles)
    psnr_input /= len(imageFiles)
    print("\n%4d PSNR | input: %.4f | predicted: %.4f\n" % (epoch, psnr_input, psnr_predict))
    return psnr_predict


def addNoise(img, sigma=12, ifInhomogeneous=False, mu=0):
    h, w, c = img.shape
    noise = np.random.normal(mu, sigma/255, (h, w, c))
    noise = np.transpose(noise, (2, 0, 1))
    if ifInhomogeneous:
        noise = np.multiply(noise,levelOfNoiseDistribution(img))
    if (img[1]==img[0]).all():
        noise[1], noise[2] = noise[0], noise[0]
    noise = np.transpose(noise, (1, 2, 0))
    return noise + img
        
def levelOfNoiseDistribution(img):
    h, w, _ = img.shape
    xi, yi = np.linspace(-w/2, w/2, w), np.linspace(-h/2, h/2, h)
    xi, yi = np.meshgrid(xi, yi)
    distribution = noiseSpatialDistributionFunction(xi, yi)/8.72
    #np.savetxt('this.txt', distribution)
    return distribution

def noiseSpatialDistributionFunction(xi, yi):
    f = np.exp(-0.0016 * (pow(xi, 2) + pow(yi, 2)) + 3)
    return f

def display_transform():
    return Compose([
        ToPILImage(),
        #Resize(400),
        #CenterCrop(400),
        ToTensor()
    ])

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def completePathImportFile(dataPath):
    files = glob.glob(os.path.join(dataPath, '*.*'))
    files.sort()
    return files

def tensorVariableNumpyImage(img):
    img = normalize(np.float32(img))
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, 0)
    img = torch.Tensor(img)
    imgCuda = Variable(img.cuda())
    return imgCuda


def dataAug(image, mode):
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    return out

def normalize(data):
    #normScale = float(1/(data.max()-data.min()))
    #x = data * normScale
    return data/255.

def patchCrop(img, scales, cropSize, stride=1):
    endc = img.shape[2]
    imgPatNum = patchCount(img, scales, cropSize, stride)
    Y = np.zeros(shape=(cropSize, cropSize, endc, imgPatNum))#, np.float32
    for k in range(len(scales)):

        img = cv2.resize(img, (int(img.shape[0]*scales[k]), int(img.shape[1]*scales[k])), interpolation=cv2.INTER_CUBIC)
        endw = img.shape[0]
        endh = img.shape[1]
        col_n = (endw - stride*2)//cropSize
        row_n = (endh - stride*2)//cropSize

        for i in range(col_n):
            for j in range(row_n):
                patch = img[stride+cropSize*i:stride+cropSize*(i+1), stride+cropSize*j:stride+cropSize*(j+1),:]
                Y[:,:,:,k] = patch
                k = k + 1
    return Y

def patchCount(img, scales, cropSize, stride=1):
    imgPatNum = 0
    for k in range(len(scales)):
        img = cv2.resize(img, (int(img.shape[0]*scales[k]), int(img.shape[1]*scales[k])), interpolation=cv2.INTER_CUBIC)
        endw = img.shape[0]
        endh = img.shape[1]
        col_n = (endw - stride*2)//cropSize
        row_n = (endh - stride*2)//cropSize
        imgPatNum += row_n * col_n
    return imgPatNum

def imgSaveTensor(img, outf, imgName):
    imgClamp = torch.clamp(img, 0., 1.)
    img= imgClamp[0].cpu()
    img= img.detach().numpy().astype(np.float32)*255
    img = np.transpose(img, (1, 2, 0))
    cv2.imwrite(os.path.join(outf, imgName), img)
    return imgClamp

def displayImage(I_A, I_B, I_C):
    fig = plt.figure()
    I_A = I_A[0,:,:].cpu()
    I_A = I_A[0].numpy().astype(np.float32)
    I_B= I_B[0,:,:].cpu()
    I_B= I_B[0].numpy().astype(np.float32)
    I_C= I_C[0,:,:].cpu()
    I_C= I_C[0].numpy().astype(np.float32)

    ax = plt.subplot("131")
    ax.imshow(I_A, cmap='gray')
    ax.set_title("Ground truth")

    ax = plt.subplot("132")
    ax.imshow(I_B, cmap='gray')
    ax.set_title("Input")

    ax = plt.subplot("133")
    ax.imshow(I_C, cmap='gray')
    ax.set_title("Model output")
    plt.show()

    return None


def batchPSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i], Img[i], data_range=data_range)
    return (PSNR/Img.shape[0])
